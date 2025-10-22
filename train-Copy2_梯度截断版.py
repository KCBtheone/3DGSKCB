#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
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
        # >>> MODULAR: 按需加载真值数据
        gt_confidence = viewpoint_cam.gt_confidence_map
        gt_normals = viewpoint_cam.gt_normal_map
        gt_curvature = viewpoint_cam.gt_curvature_map
        bg = torch.rand(3, device="cuda") if opt.random_background else background
        torch.cuda.synchronize(); time_accumulators["data_loading"] += time.time() - time_start

        # --- 渲染 ---
        torch.cuda.synchronize(); time_start = time.time()
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        torch.cuda.synchronize(); time_accumulators["render"] += time.time() - time_start

        # --- 损失计算 ---
        torch.cuda.synchronize(); time_start = time.time()
        mean_normal_error = torch.tensor(0.0, device="cuda")
        mean_curvature_error = torch.tensor(0.0, device="cuda")

        # ========================================================================
        # A. L1损失的动态权重计算 (MODULARIZED)
        # ========================================================================
        
        # --- 初始化所有权重为中性值 ---
        W_C = torch.tensor(1.0, device="cuda") # 置信度权重项
        W_N = torch.tensor(1.0, device="cuda") # 法线几何关注权重
        W_I = torch.tensor(1.0, device="cuda") # 浮游物关注权重
        
        # --- 步骤1: 计算置信度权重 (W_C) ---
        if gt_confidence is not None and opt.confidence_loss_type != 'none':
            conf_map = gt_confidence.unsqueeze(0)
            if opt.confidence_loss_type == 'multiplicative':
                W_C = torch.pow(conf_map, opt.confidence_gamma)
            elif opt.confidence_loss_type == 'inverse_multiplicative':
                W_C = 1.0 + opt.confidence_alpha_fix * torch.pow(1.0 - conf_map, opt.confidence_gamma_fix)

        # --- 步骤2: 计算法线引导权重 (W_N) ---
        use_geometry_guidance = (iteration >= opt.geometry_start_iter)
        rendered_normals = None # 预先定义, 以便SSIM部分可以使用
        
        # 仅当开关打开、达到起始迭代、且数据存在时, 才计算法线权重
        if opt.use_normal_guidance and use_geometry_guidance and gt_normals is not None:
            # [关键修正] 移除 torch.no_grad() 以允许梯度回传到几何参数
            # 我们需要法线误差的梯度来优化高斯基元的旋转和缩放
            normals_for_render = (gaussians.get_normals + 1.0) / 2.0
            normal_render_pkg = render(viewpoint_cam, gaussians, pipe, bg, override_color=normals_for_render)
            rendered_normals = normal_render_pkg["render"]
            
            gt_normals_cuda = gt_normals.cuda()
            mask = (gt_normals_cuda.sum(dim=0) != 0)
            if mask.sum() > 0:
                # 计算法线误差图 (cosine distance)
                error_map = 1.0 - torch.sum(rendered_normals * gt_normals_cuda, dim=0)
                mean_normal_error = error_map[mask].mean().detach() # 日志记录用 detach
                
                # 如果alpha > 0, 则根据误差图生成权重
                if opt.alpha_normals > 0:
                    W_N = 1.0 + opt.alpha_normals * error_map.unsqueeze(0)

        # --- 步骤3: 计算浮游物权重 (W_I) ---
        if opt.use_isotropy_loss and iteration >= opt.isotropy_start_iter:
            # 假设get_isotropy是可导的，这样梯度才能回传
            isotropy_indices = gaussians.get_isotropy.expand(-1, 3)
            isotropy_map = render(viewpoint_cam, gaussians, pipe, torch.zeros_like(bg), override_color=isotropy_indices)["render"][:1]
            if opt.lambda_isotropy > 0:
                W_I = 1.0 + opt.lambda_isotropy * isotropy_map

        # --- 步骤4: 最终组合权重并计算L1损失 ---
        if opt.confidence_loss_type == 'additive':
            # 方案三: 加法组合 W_L1 = 1 + W'_C + W'_N + W'_I
            W_C_prime = torch.tensor(0.0, device="cuda")
            if gt_confidence is not None:
                 W_C_prime = opt.confidence_alpha_fix * torch.pow(1.0 - gt_confidence.unsqueeze(0), opt.confidence_gamma_fix)
            
            W_N_prime = W_N - 1.0 # 提取增量部分
            W_I_prime = W_I - 1.0 # 提取增量部分
            final_l1_weight_map = 1.0 + W_C_prime + W_N_prime + W_I_prime
        else:
            # 方案一、二和'none'模式都使用乘法结构
            final_l1_weight_map = W_C * W_N * W_I

        # detach()已移除，权重图现在会贡献梯度
        l1_loss_val = (final_l1_weight_map * torch.abs(image - gt_image)).mean()

        # ========================================================================
        # B. SSIM损失计算 (Standard vs. Structure-Aware)
        # ========================================================================
        # 仅当SA-SSIM开关打开、达到起始迭代、且所有必需数据(法线图/曲率图)都存在时, 才使用它
        can_use_sa_ssim = (opt.use_sa_ssim and use_geometry_guidance and 
                           gt_curvature is not None and rendered_normals is not None and
                           'mask' in locals() and mask.sum() > 0)

        if not can_use_sa_ssim:
            ssim_loss_val = 1.0 - ssim(image, gt_image)
        else:
            # [关键修正] 移除 with torch.no_grad()
            gt_curvature_cuda = gt_curvature.cuda()
            sobel_kernel = torch.tensor([[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]], dtype=torch.float32, device="cuda").reshape(1,1,3,3).repeat(3,1,1,1)
            
            # 计算渲染法线图的曲率, 需要梯度
            rendered_curvature = F.conv2d(rendered_normals.unsqueeze(0), sobel_kernel, padding=1, groups=3).squeeze(0).pow(2).sum(dim=0).sqrt()
            curvature_error_map = torch.abs(rendered_curvature - gt_curvature_cuda[0])
            mean_curvature_error = curvature_error_map[mask].mean().detach() # 日志记录用 detach
            
            # 'error_map' 来自上面的法线计算部分，它已经包含了梯度
            M_geo = (1 - opt.beta_geo) * error_map + opt.beta_geo * curvature_error_map
            
            current_gamma = opt.gamma_base
            if opt.adaptive_gamma:
                if iteration < opt.gamma_warmup:
                    current_gamma *= (iteration / opt.gamma_warmup)
                else:
                    valid_geo_errors = M_geo[mask]
                    if valid_geo_errors.numel() > 1:
                        # 限制标准差, 防止爆炸。计算标准差时阻断梯度以增加稳定性
                        std_dev = torch.std(valid_geo_errors.detach())
                        current_gamma *= torch.clamp(std_dev, 0.0, 2.0)
            
            # 使用EMA平滑gamma, 防止剧烈波动。EMA更新也使用detach防止复合梯度
            gamma_ema = 0.99 * gamma_ema.detach() + 0.01 * current_gamma
            W_geo = 1.0 + gamma_ema * M_geo.unsqueeze(0)
            
            ssim_loss_val = 1.0 - ssim_weighted(image, gt_image, W_geo)

        total_loss = (1.0 - opt.lambda_dssim) * l1_loss_val + opt.lambda_dssim * ssim_loss_val
        torch.cuda.synchronize(); time_accumulators["loss_calc"] += time.time() - time_start

        # --- 反向传播 ---
        torch.cuda.synchronize(); time_start = time.time()
        total_loss.backward()
        torch.cuda.synchronize(); time_accumulators["backward"] += time.time() - time_start

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
        torch.cuda.synchronize(); time_accumulators["optimizer_densify"] += time.time() - time_start
        
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
                    print(f"\n[ITER {iteration}] ✨ New best model found! PSNR: {best_test_psnr:.2f}. Saving 'best.ply'...")
                    scene.save(iteration, is_best=True)

            # 固定的训练检查点保存
            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving training checkpoint...")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # CSV与Tensorboard日志
            log_data = {
                'Iteration': iteration, 'Total_Loss': total_loss.item(), 'L1_Loss': l1_loss_val.item(),
                'SSIM_Loss': ssim_loss_val.item(), 'Train_PSNR': current_psnr, 'Test_PSNR': last_test_psnr,
                'Total_Points': gaussians.get_xyz.shape[0], 'Total_Iter_Time_s': iter_total_end_time - iter_total_start_time,
                'Normal_Error_Mean': mean_normal_error.item(), 'Curvature_Error_Mean': mean_curvature_error.item(),
                'Adaptive_Gamma': gamma_ema.item()
            }
            log_to_csv(log_data)
            
            if TENSORBOARD_FOUND and tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('train_loss/total_loss', total_loss.item(), iteration)
                tb_writer.add_scalar('train_loss/l1_loss', l1_loss_val.item(), iteration)
                tb_writer.add_scalar('train_loss/ssim_loss', ssim_loss_val.item(), iteration)
                tb_writer.add_scalar('metrics/train_psnr', current_psnr, iteration)
                tb_writer.add_scalar('metrics/test_psnr', last_test_psnr, iteration)
                tb_writer.add_scalar('gaussians/count', gaussians.get_xyz.shape[0], iteration)
                tb_writer.add_scalar('params/learning_rate', gaussians.optimizer.param_groups[0]['lr'], iteration)
                tb_writer.add_scalar('params/adaptive_gamma', gamma_ema.item(), iteration)
            
            # 计时报告
            if iteration % 1000 == 0 and iteration > first_iter:
                num_iters_since_report = iteration - last_report_iter
                if num_iters_since_report > 0:
                    avg_times_ms = {k: v / num_iters_since_report * 1000 for k, v in time_accumulators.items()}
                    total_reported_time_ms = sum(v for k, v in avg_times_ms.items() if k != "total_iteration")
                    print("\n" + "="*80)
                    print(f"--- [ Timing Report - Iteration {iteration} ] ---")
                    print(f"  - Data Loading:        {avg_times_ms['data_loading']:>8.2f} ms")
                    print(f"  - Rendering (Forward):   {avg_times_ms['render']:>8.2f} ms")
                    print(f"  - Loss Calculation:    {avg_times_ms['loss_calc']:>8.2f} ms")
                    print(f"  - Backward Pass:       {avg_times_ms['backward']:>8.2f} ms")
                    print(f"  - Optimizer & Densify: {avg_times_ms['optimizer_densify']:>8.2f} ms")
                    print(f"  -------------------------------------------")
                    print(f"  Sum of parts:          {total_reported_time_ms:>8.2f} ms")
                    print(f"  Total Iteration Time:  {avg_times_ms['total_iteration']:>8.2f} ms")
                    print("="*80 + "\n")
                    for key in time_accumulators: time_accumulators[key] = 0.0
                    last_report_iter = iteration

    progress_bar.close()
    print(f"\nTraining complete. Final best Test PSNR: {best_test_psnr:.2f}")
    close_csv_logger()

def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[0:10])
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None
    return tb_writer

@torch.no_grad()
def validation_report(tb_writer, iteration, scene, render_fn, render_args):
    l1_test, psnr_test = 0.0, 0.0
    test_cameras = scene.getTestCameras()
    if not test_cameras:
        print("No test cameras found. Skipping validation.")
        return 0.0, 0.0
    
    for viewpoint in tqdm(test_cameras, desc="Evaluating [test]"):
        image = torch.clamp(render_fn(viewpoint, scene.gaussians, *render_args)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        l1_test += l1_loss(image, gt_image).item()
        psnr_test += psnr(image, gt_image).mean().item()
        
    psnr_test /= len(test_cameras)
    l1_test /= len(test_cameras)
    
    if TENSORBOARD_FOUND and tb_writer:
        tb_writer.add_scalar('validation/l1_loss', l1_test, iteration)
        tb_writer.add_scalar('validation/psnr', psnr_test, iteration)
        
    print(f"\n[ITER {iteration}] Validation Results: L1 {l1_test:.5f} PSNR {psnr_test:.2f}")
    return psnr_test, l1_test

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[]) # 将由代码动态生成
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[]) # 将由代码动态生成
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    args = get_combined_args(parser)
    
    # [ 关键修正 ] 确保 'start_checkpoint' 属性总是存在, 避免 AttributeError
    # 这一块代码的缩进必须和上面及下面的代码行对齐
    if not hasattr(args, 'start_checkpoint'):
        args.start_checkpoint = None

    # 确保在恢复训练时也能正确引用opt_params
    opt_params = op.extract(args)
    # 将iterations从opt_params同步回args, 以便其他部分(如动态迭代计划)可以访问
    args.iterations = opt_params.iterations
    
    # --- 动态生成测试和保存计划 ---
    saving_iters = set(args.save_iterations)
    saving_iters.add(7000)
    saving_iters.add(args.iterations)
    args.save_iterations = sorted(list(saving_iters))
    
    testing_iters = set(args.test_iterations)
    # 默认从7000开始, 每1000次测试一次, 直到结束
    for i in range(7000, args.iterations + 1, 1000):
        testing_iters.add(i)
    args.test_iterations = sorted(list(testing_iters))
    
    print("Training with args:\n" + str(args))
    print(f"Milestone save iterations: {args.save_iterations}")
    print(f"Testing iterations: {args.test_iterations}")
    
    safe_state(False)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), opt_params, pp.extract(args), 
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
             args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")