#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, ssim_weighted
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
import logging
import time
import csv
import torch.nn.functional as F
import torchvision

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# =================================================================================
# >>> [ 日志系统 ] <<<
# =================================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG_FILE_HANDLE = None
CSV_WRITER = None

def setup_csv_logger(model_path):
    global LOG_FILE_HANDLE, CSV_WRITER
    log_file = os.path.join(model_path, "training_log.csv")
    fieldnames = [
        'Iteration', 'Total_Loss', 'L1_Loss', 'SSIM_Loss',
        'Train_PSNR', 'Test_PSNR', 'Total_Points', 'Total_Iter_Time_s',
        'Normal_Error_Mean', 'Curvature_Error_Mean', 'Adaptive_Gamma'
    ]
    try:
        LOG_FILE_HANDLE = open(log_file, 'w', newline='')
        CSV_WRITER = csv.DictWriter(LOG_FILE_HANDLE, fieldnames=fieldnames)
        CSV_WRITER.writeheader()
        logging.info(f"CSV training log will be saved to: {log_file}")
    except IOError as e:
        logging.error(f"Failed to create CSV log file {log_file}. Reason: {e}")

def log_to_csv(data_dict):
    global CSV_WRITER, LOG_FILE_HANDLE
    if CSV_WRITER:
        default_data = {key: "N/A" for key in CSV_WRITER.fieldnames}
        default_data.update(data_dict)
        CSV_WRITER.writerow(default_data)
        if data_dict.get('Iteration', 0) % 100 == 0 and LOG_FILE_HANDLE:
            LOG_FILE_HANDLE.flush()

def close_csv_logger():
    global LOG_FILE_HANDLE
    if LOG_FILE_HANDLE:
        LOG_FILE_HANDLE.close()
        logging.info("CSV training log file closed.")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, start_checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    if start_checkpoint:
        (model_params, first_iter) = torch.load(start_checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Resuming training from iteration {first_iter}")

    # --- 初始化核心变量 ---
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # [新增] 追踪最佳测试PSNR和EMA日志变量
    best_test_psnr = 0.0
    last_test_psnr = 0.0
    gamma_ema = torch.tensor(0.0, device="cuda")

    # [新增] 精确计时器
    time_accumulators = {k: 0.0 for k in ["data_loading", "render", "loss_calc", "backward", "optimizer_densify", "total_iteration"]}
    last_report_iter = first_iter

    setup_csv_logger(dataset.model_path)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress")

    for iteration in progress_bar:
        
        torch.cuda.synchronize(); iter_total_start_time = time.time()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # --- 数据加载 ---
        torch.cuda.synchronize(); time_start = time.time()
        viewpoint_cam = scene.getTrainCameras()[randint(0, len(scene.getTrainCameras()) - 1)]
        gt_image = viewpoint_cam.original_image.cuda()
        gt_normals = viewpoint_cam.gt_normal_map
        gt_curvature = viewpoint_cam.gt_curvature_map
        bg = torch.rand(3, device="cuda") if opt.random_background else background
        torch.cuda.synchronize(); time_accumulators["data_loading"] += time.time() - time.time()

        # --- 渲染 ---
        torch.cuda.synchronize(); time_start = time.time()
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        torch.cuda.synchronize(); time_accumulators["render"] += time.time() - time.time()

        # --- 损失计算 ---
        torch.cuda.synchronize(); time_start = time.time()
        mean_normal_error = torch.tensor(0.0, device="cuda")
        mean_curvature_error = torch.tensor(0.0, device="cuda")

        # A. L1损失的动态权重计算
        gt_confidence = viewpoint_cam.gt_confidence_map
        
        # [ 🚀 实验性修改: 方案二 (train_2.py) 🚀 ]
        # 原始逻辑 (train.py):
        # final_l1_weight_map = torch.ones_like(image[:1]) if gt_confidence is None else torch.pow(gt_confidence.unsqueeze(0), opt.confidence_gamma)
        
        # 新逻辑 (方案二): 反转置信度，"关注"低置信度区域
        if gt_confidence is None:
            final_l1_weight_map = torch.ones_like(image[:1])
        else:
            # 硬编码新参数
            alpha_fix = 1.0 
            gamma_fix = 1.0 # (gamma_fix=1.0 意味着 pow 操作是可选的, 但保留结构)
            
            # W_C(p) = 1 + alpha_fix * (1 - M_conf(p))^gamma_fix
            inv_conf = 1.0 - gt_confidence.unsqueeze(0)
            final_l1_weight_map = 1.0 + alpha_fix * torch.pow(inv_conf, gamma_fix)
            # [注意] 原有的 opt.confidence_gamma (gamma_conf) 在此方案中被弃用
        # [ 🚀 实验性修改结束 🚀 ]

        use_normal_guidance = (iteration >= opt.geometry_start_iter and gt_normals is not None)
        rendered_normals = None
        if use_normal_guidance:
            normals_for_render = (gaussians.get_normals + 1.0) / 2.0
            normal_render_pkg = render(viewpoint_cam, gaussians, pipe, bg, override_color=normals_for_render)
            rendered_normals = normal_render_pkg["render"]
            gt_normals_cuda = gt_normals.cuda()
            mask = (gt_normals_cuda.sum(dim=0) != 0)
            if mask.sum() > 0:
                error_map = 1.0 - torch.sum(rendered_normals * gt_normals_cuda, dim=0)
                mean_normal_error = error_map[mask].mean()
                if opt.alpha_normals > 0:
                    final_l1_weight_map *= (1.0 + opt.alpha_normals * error_map).unsqueeze(0)
        
        if opt.lambda_isotropy > 0 and iteration >= opt.isotropy_start_iter:
            isotropy_indices = gaussians.get_isotropy.detach().expand(-1, 3)
            isotropy_map = render(viewpoint_cam, gaussians, pipe, torch.zeros_like(bg), override_color=isotropy_indices)["render"][:1]
            final_l1_weight_map *= (1.0 + opt.lambda_isotropy * isotropy_map)

        l1_loss_val = (final_l1_weight_map * torch.abs(image - gt_image)).mean()

        # B. SSIM损失计算 (原始 vs. 结构感知)
        if not opt.use_sa_ssim or not use_normal_guidance or gt_curvature is None or rendered_normals is None:
            ssim_loss_val = 1.0 - ssim(image, gt_image)
        else:
            with torch.no_grad():
                gt_curvature_cuda = gt_curvature.cuda()
                sobel_kernel = torch.tensor([[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]], dtype=torch.float32, device="cuda").reshape(1,1,3,3).repeat(3,1,1,1)
                rendered_curvature = F.conv2d(rendered_normals.unsqueeze(0), sobel_kernel, padding=1, groups=3).squeeze(0).pow(2).sum(dim=0).sqrt()
EOM
                curvature_error_map = torch.abs(rendered_curvature - gt_curvature_cuda[0])
                mean_curvature_error = curvature_error_map[mask].mean()
                
                M_geo = (1 - opt.beta_geo) * error_map + opt.beta_geo * curvature_error_map
                
                current_gamma = opt.gamma_base
                if opt.adaptive_gamma:
                    if iteration < opt.gamma_warmup:
                        current_gamma *= (iteration / opt.gamma_warmup)
                    else:
                        valid_geo_errors = M_geo[mask]
                        if valid_geo_errors.numel() > 1:
                            current_gamma *= torch.clamp(torch.std(valid_geo_errors), 0.0, 2.0)
                gamma_ema = 0.99 * gamma_ema + 0.01 * current_gamma
                W_geo = 1.0 + gamma_ema * M_geo.unsqueeze(0)
            ssim_loss_val = 1.0 - ssim_weighted(image, gt_image, W_geo)

        total_loss = (1.0 - opt.lambda_dssim) * l1_loss_val + opt.lambda_dssim * ssim_loss_val
        torch.cuda.synchronize(); time_accumulators["loss_calc"] += time.time() - time.time()

        # --- 反向传播 ---
        torch.cuda.synchronize(); time_start = time.time()
        total_loss.backward()
        torch.cuda.synchronize(); time_accumulators["backward"] += time.time() - time.time()

        # --- 优化与致密化 ---
        torch.cuda.synchronize(); time_start = time.time()
        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor.grad, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        torch.cuda.synchronize(); time_accumulators["optimizer_densify"] += time.time() - time.time()
        
        torch.cuda.synchronize(); iter_total_end_time = time.time()
        time_accumulators["total_iteration"] += iter_total_end_time - iter_total_start_time

        # --- 日志、评估与保存 ---
        with torch.no_grad():
            current_psnr = psnr(image, gt_image).mean().item()
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.5f}",
                "PSNR": f"{current_psnr:.2f}",
                "Gaussians": f"{gaussians.get_xyz.shape[0]}"
            })

            # 固定的里程碑保存
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving milestone Gaussians...")
                scene.save(iteration)

            # 周期性的测试与最佳模型保存
            if (iteration in testing_iterations):
                print(f"\n[ITER {iteration}] Testing...")
                current_test_psnr, _ = validation_report(None, iteration, scene, render, (pipe, background))
                last_test_psnr = current_test_psnr
                
                if current_test_psnr > best_test_psnr:
                    best_test_psnr = current_test_psnr
      _message_num=2, truncated=True