#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, ssim_weighted, VGGPerceptualLoss, StructuralLoss, SynergyGuidance
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

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# =================================================================================
#  日志系统 
# =================================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG_FILE_HANDLE = None
CSV_WRITER = None

def setup_csv_logger(model_path):
    global LOG_FILE_HANDLE, CSV_WRITER
    log_file = os.path.join(model_path, "training_log.csv")
    fieldnames = [
        'Iteration', 'Total_Loss', 'L1_Loss', 'SSIM_Loss', 'Perceptual_Loss', 'Geo_Loss', 'Smooth_Loss', 'Struct_Loss',
        'L1_High_Conf', 'L1_Low_Conf',
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

    # --- [ V5 Framework Initialization ] ---
    perceptual_loss_fn = None
    if opt.use_perceptual_loss:
        print("INFO: Perceptual loss enabled.")
        perceptual_loss_fn = VGGPerceptualLoss().to("cuda")

    structural_loss_fn = None
    if opt.structural_loss_mode != 'none':
        print(f"INFO: [Diagnostics] Structural loss mode '{opt.structural_loss_mode}' enabled.")
        structural_loss_fn = StructuralLoss(opt).to("cuda")

    synergy_guidance_fn = SynergyGuidance(opt).to("cuda")
    if opt.structural_loss_mode != 'none':
        print(f"INFO: [Guidance] Synergy mode '{opt.synergy_mode}' enabled.")

    if start_checkpoint:
        (model_params, first_iter) = torch.load(start_checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Resuming training from iteration {first_iter}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    best_test_psnr = 0.0
    last_test_psnr = 0.0
    gamma_ema = torch.tensor(0.0, device="cuda")

    time_accumulators = {k: 0.0 for k in ["data_loading", "render", "loss_calc", "backward", "optimizer_densify", "total_iteration"]}
    last_report_iter = first_iter

    setup_csv_logger(dataset.model_path)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress")

    for iteration in progress_bar:

        torch.cuda.synchronize(); iter_total_start_time = time.time()
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        torch.cuda.synchronize(); time_start = time.time()
        
        viewpoint_cam = scene.getTrainCameras()[randint(0, len(scene.getTrainCameras()) - 1)]
        
        gt_image = viewpoint_cam.original_image.cuda()
        if gt_image.shape[0] == 4:
            gt_image = gt_image[:3, ...] * gt_image[3:4, ...] + (1.0 - gt_image[3:4, ...]) * background.unsqueeze(1).unsqueeze(2)

        gt_confidence = getattr(viewpoint_cam, 'gt_confidence_map', None)
        gt_normals = getattr(viewpoint_cam, 'gt_normal_map', None)
        gt_curvature = getattr(viewpoint_cam, 'gt_curvature_map', None)
        bg = torch.rand(3, device="cuda") if opt.random_background else background
        torch.cuda.synchronize(); time_accumulators["data_loading"] += time.time() - time_start

        torch.cuda.synchronize(); time_start = time.time()
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        torch.cuda.synchronize(); time_accumulators["render"] += time.time() - time_start

        torch.cuda.synchronize(); time_start = time.time()

        densify_grads = None
        if opt.decouple_densification_grad and iteration < opt.densify_until_iter:
            l1_for_densify = l1_loss(image, gt_image)
            densify_grads = torch.autograd.grad(
                outputs=l1_for_densify, inputs=viewspace_point_tensor,
                grad_outputs=torch.ones_like(l1_for_densify), create_graph=False, retain_graph=True
            )[0]

        l1_loss_val, ssim_loss_val, geo_loss, smooth_loss, perceptual_loss, struct_loss = [torch.tensor(0.0, device="cuda") for _ in range(6)]
        loss_l1_high, loss_l1_low = [torch.tensor(0.0, device="cuda") for _ in range(2)]
        mean_normal_error, mean_curvature_error = [torch.tensor(0.0, device="cuda") for _ in range(2)]

        W_N = torch.tensor(1.0, device="cuda")
        W_I = torch.tensor(1.0, device="cuda")
        use_geometry_guidance = (iteration >= opt.geometry_start_iter)
        rendered_normals = None
        if opt.use_normal_guidance and use_geometry_guidance and gt_normals is not None:
            normals_for_render = (gaussians.get_normals + 1.0) / 2.0
            rendered_normals = render(viewpoint_cam, gaussians, pipe, bg, override_color=normals_for_render)["render"]
            gt_normals_cuda = gt_normals.cuda()
            mask = (gt_normals_cuda.sum(dim=0) != 0)
            if mask.sum() > 0:
                error_map = 1.0 - torch.sum(rendered_normals * gt_normals_cuda, dim=0)
                mean_normal_error = error_map[mask].mean().detach()
                if opt.alpha_normals > 0: W_N = 1.0 + opt.alpha_normals * error_map.unsqueeze(0)
        if opt.use_isotropy_loss and iteration >= opt.isotropy_start_iter:
            isotropy_indices = gaussians.get_isotropy.expand(-1, 3)
            isotropy_map = render(viewpoint_cam, gaussians, pipe, torch.zeros_like(bg), override_color=isotropy_indices)["render"][:1]
            if opt.lambda_isotropy > 0: W_I = 1.0 + opt.lambda_isotropy * isotropy_map
        if 'mask' in locals() and mask.sum() > 0 and opt.lambda_normals > 0 and 'error_map' in locals():
            geo_loss = error_map[mask].mean()

        feedback_map = None
        if structural_loss_fn is not None and iteration >= opt.struct_loss_start_iter:
            struct_loss, feedback_map = structural_loss_fn(image, gt_image)

        W_S, guided_ssim_loss = synergy_guidance_fn(feedback_map, image, gt_image)

        C = gt_confidence.unsqueeze(0) if gt_confidence is not None else None
        l1_error_map = torch.abs(image - gt_image)

        if opt.confidence_scheme == 'multiplicative':
            W_C = torch.pow(C, opt.confidence_gamma) if C is not None else torch.tensor(1.0, device="cuda")
            W_total = W_C * W_N * W_I * W_S
            l1_loss_val = (W_total.detach() * l1_error_map).mean()
            geo_loss = torch.tensor(0.0, device="cuda")
        elif opt.confidence_scheme == 'dual_l1':
            if C is not None:
                mask_high_conf = (C > opt.confidence_thresh).float()
                mask_low_conf = 1.0 - mask_high_conf
                loss_l1_high = (l1_error_map * mask_high_conf * W_S.detach()).sum() / (mask_high_conf.sum() + 1e-8)
                loss_l1_low = (l1_error_map * mask_low_conf).sum() / (mask_low_conf.sum() + 1e-8)
                l1_loss_val = loss_l1_high + opt.lambda_low_confidence * loss_l1_low
            else:
                l1_loss_val = (l1_error_map * W_S.detach()).mean()
        else:
            l1_loss_val = (l1_error_map * W_S.detach()).mean()
            if opt.confidence_scheme == 'gatekeeper':
                 avg_C = C.mean().item() if C is not None else 1.0
                 l1_loss_val = l1_loss_val * (avg_C**opt.confidence_gamma)
            elif opt.confidence_scheme == 'arbitrator':
                C_map = C if C is not None else torch.ones_like(image[:1])
                geo_prior_map = W_N - 1.0
                photometric_term = (C_map**opt.confidence_gamma * l1_error_map * W_S.detach()).mean()
                geometric_term = ((1.0 - C_map)**opt.confidence_gamma * geo_prior_map).mean() * opt.lambda_geo_low_conf
                l1_loss_val = photometric_term
                geo_loss = geometric_term

        if guided_ssim_loss is not None:
            ssim_loss_val = 1.0 - guided_ssim_loss
        else:
            can_use_sa_ssim = (opt.use_sa_ssim and use_geometry_guidance and gt_curvature is not None and rendered_normals is not None and 'mask' in locals() and mask.sum() > 0)
            if not can_use_sa_ssim:
                ssim_loss_val = 1.0 - ssim(image, gt_image)
            else:
                gt_curvature_cuda = gt_curvature.cuda()
                sobel_kernel = torch.tensor([[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]], dtype=torch.float32, device="cuda").reshape(1,1,3,3).repeat(3,1,1,1)
                rendered_curvature = F.conv2d(rendered_normals.unsqueeze(0), sobel_kernel, padding=1, groups=3).squeeze(0).pow(2).sum(dim=0).sqrt()
                curvature_error_map = torch.abs(rendered_curvature - gt_curvature_cuda[0])
                mean_curvature_error = curvature_error_map[mask].mean().detach()
                M_geo = (1 - opt.beta_geo) * error_map + opt.beta_geo * curvature_error_map
                current_gamma = opt.gamma_base
                if opt.adaptive_gamma:
                    if iteration < opt.gamma_warmup: current_gamma *= (iteration / opt.gamma_warmup)
                    else:
                        valid_geo_errors = M_geo[mask]
                        if valid_geo_errors.numel() > 1: current_gamma *= torch.clamp(torch.std(valid_geo_errors.detach()), 0.0, 2.0)
                gamma_ema = 0.99 * gamma_ema.detach() + 0.01 * current_gamma
                W_ssim_geo = 1.0 + gamma_ema * M_geo.unsqueeze(0)
                ssim_loss_val = 1.0 - ssim_weighted(image, gt_image, W_ssim_geo)

        if perceptual_loss_fn and iteration >= opt.perceptual_start_iter:
            perceptual_loss = perceptual_loss_fn(image, gt_image)
        if opt.use_smoothness_loss and iteration >= opt.smooth_start_iter:
            smooth_loss = gaussians.compute_smoothness_loss()

        total_loss = (1.0 - opt.lambda_dssim) * l1_loss_val + \
                     opt.lambda_dssim * ssim_loss_val + \
                     opt.lambda_perceptual * perceptual_loss + \
                     opt.lambda_normals * geo_loss + \
                     opt.lambda_smooth * smooth_loss + \
                     opt.lambda_struct_loss * struct_loss

        torch.cuda.synchronize(); time_accumulators["loss_calc"] += time.time() - time_start
        torch.cuda.synchronize(); time_start = time.time()
        total_loss.backward()
        torch.cuda.synchronize(); time_accumulators["backward"] += time.time() - time_start
        torch.cuda.synchronize(); time_start = time.time()
        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                grad_for_densify = densify_grads if opt.decouple_densification_grad and densify_grads is not None else viewspace_point_tensor.grad
                gaussians.add_densification_stats(grad_for_densify, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt, iteration, scene.cameras_extent, size_threshold)
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); time_accumulators["optimizer_densify"] += time.time() - time_start
        torch.cuda.synchronize(); iter_total_end_time = time.time()
        time_accumulators["total_iteration"] += iter_total_end_time - iter_total_start_time

        with torch.no_grad():
            current_psnr = psnr(image, gt_image).mean().item()
            progress_bar.set_postfix({ "DiagMode": opt.structural_loss_mode, "GuideMode": opt.synergy_mode, "PSNR": f"{current_psnr:.2f}" })

            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving milestone...")
                scene.save(iteration)
            if (iteration in testing_iterations):
                print(f"\n[ITER {iteration}] Testing...")
                current_test_psnr, _ = validation_report(tb_writer, iteration, scene, render, (pipe, background))
                last_test_psnr = current_test_psnr
                if current_test_psnr > best_test_psnr:
                    best_test_psnr = current_test_psnr
                    print(f"\n[ITER {iteration}] ✨ New best model! PSNR: {best_test_psnr:.2f}. Saving...")
                    scene.save(iteration, is_best=True)
            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving checkpoint...")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            log_data = {
                'Iteration': iteration, 'Total_Loss': total_loss.item(), 'L1_Loss': l1_loss_val.item(),
                'SSIM_Loss': ssim_loss_val.item(), 'Perceptual_Loss': perceptual_loss.item(),
                'Geo_Loss': geo_loss.item(), 'Smooth_Loss': smooth_loss.item(), 'Struct_Loss': struct_loss.item(),
                'L1_High_Conf': loss_l1_high.item(), 'L1_Low_Conf': loss_l1_low.item(),
                'Train_PSNR': current_psnr, 'Test_PSNR': last_test_psnr,
                'Total_Points': gaussians.get_xyz.shape[0], 'Total_Iter_Time_s': iter_total_end_time - iter_total_start_time,
                'Normal_Error_Mean': mean_normal_error.item(), 'Curvature_Error_Mean': mean_curvature_error.item(),
                'Adaptive_Gamma': gamma_ema.item()
            }
            log_to_csv(log_data)
            
            if TENSORBOARD_FOUND and tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('train_loss/total', total_loss.item(), iteration)
                tb_writer.add_scalar('train_loss/l1', l1_loss_val.item(), iteration)
                tb_writer.add_scalar('train_loss/ssim', ssim_loss_val.item(), iteration)
                tb_writer.add_scalar('train_loss/struct', struct_loss.item(), iteration)
                tb_writer.add_scalar('metrics/train_psnr', current_psnr, iteration)
                tb_writer.add_scalar('metrics/test_psnr', last_test_psnr, iteration)
                tb_writer.add_scalar('gaussians/count', gaussians.get_xyz.shape[0], iteration)

    progress_bar.close()
    print("\nTraining complete.")
    close_csv_logger()

def prepare_output_and_logger(args):
    if not args.model_path:
        diag_str = args.structural_loss_mode if args.structural_loss_mode != 'none' else 'base'
        guide_str = args.synergy_mode if diag_str != 'base' else 'nosyn'
        folder_name = f"{diag_str}_{guide_str}_{str(uuid.uuid4())[0:6]}"
        args.model_path = os.path.join("./output/", folder_name)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    return SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None

@torch.no_grad()
def validation_report(tb_writer, iteration, scene, render_fn, render_args):
    l1_test, psnr_test = 0.0, 0.0
    test_cameras = scene.getTestCameras()
    if not test_cameras: return 0.0, 0.0

    for viewpoint in tqdm(test_cameras, desc="Evaluating [test]"):
        image = torch.clamp(render_fn(viewpoint, scene.gaussians, *render_args)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
        
        if gt_image.shape[0] == 4:
            gt_image = gt_image[:3] * gt_image[3:4] + (1 - gt_image[3:4]) * render_args[1].unsqueeze(1).unsqueeze(2)

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    args = get_combined_args(parser)
    args.iterations = op.extract(args).iterations


    arg_defaults = {
        'test_iterations': [], 'save_iterations': [], 'checkpoint_iterations': [],
        'start_checkpoint': None, 'debug_from': -1, 'detect_anomaly': False
    }
    for arg, default_val in arg_defaults.items():
        if not hasattr(args, arg):
            setattr(args, arg, default_val)

    if hasattr(args, 'structural_loss_mode'):
        mode_map = {'base_grad': 'sobel', 'ms_grad': 'ms_sobel', 'struct': 'struct_ssim'}
        if args.structural_loss_mode in mode_map:
            new_mode = mode_map[args.structural_loss_mode]
            print(f"Backward compatibility: Remapping legacy mode '{args.structural_loss_mode}' to '{new_mode}'.")
            args.structural_loss_mode = new_mode
    if hasattr(args, 'lambda_grad') and args.lambda_grad > 0 and args.structural_loss_mode == 'none':
        print(f"Backward compatibility: Detected legacy 'lambda_grad'={args.lambda_grad}. Activating structural loss with mode 'sobel'.")
        args.structural_loss_mode = 'sobel'
        args.lambda_struct_loss = args.lambda_grad
    if hasattr(args, 'alpha_struct_feedback') and args.alpha_struct_feedback is not None and args.synergy_mode == 'v5_ultimate':
        val = args.alpha_struct_feedback
        print(f"Backward compatibility: Detected 'alpha_struct_feedback'={val} with 'v5_ultimate' mode. Mapping it to alpha_l1 and alpha_ssim.")
        args.alpha_l1_feedback = val
        args.alpha_ssim_feedback = val

    saving_iters = set(args.save_iterations)
    testing_iters = set(args.test_iterations)
    if args.iterations not in saving_iters: saving_iters.add(args.iterations)
    if args.iterations not in testing_iters: testing_iters.add(args.iterations)

    print("Training with final processed args:\n" + str(args))
    print(f"Milestone save iterations: {sorted(list(saving_iters))}")
    print(f"Testing iterations: {sorted(list(testing_iters))}")

    safe_state(False)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args),
             sorted(list(testing_iters)),
             sorted(list(saving_iters)),
             sorted(list(set(args.checkpoint_iterations))),
             args.start_checkpoint,
             args.debug_from)

    print("\nTraining complete.")