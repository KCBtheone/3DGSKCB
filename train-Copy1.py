# train.py

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
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

# Setup a simple logger for clean console output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CSV 日志记录器 ---
LOG_FILE_HANDLE = None
CSV_WRITER = None

def setup_csv_logger(model_path):
    """初始化CSV日志记录器"""
    global LOG_FILE_HANDLE, CSV_WRITER
    log_file = os.path.join(model_path, "training_log.csv")
    # 定义CSV文件的表头 (已更新)
    fieldnames = ['Iteration', 'Total_Loss', 'L1_Loss', 'SSIM_Loss', 
                  'Line_Loss(part)', 'Normal_Loss(part)', 'Depth_Loss(part)',
                  'Train_PSNR', 'Total_Points', 'Total_Iter_Time_s']
    
    try:
        LOG_FILE_HANDLE = open(log_file, 'w', newline='')
        CSV_WRITER = csv.DictWriter(LOG_FILE_HANDLE, fieldnames=fieldnames)
        CSV_WRITER.writeheader()
        logging.info(f"CSV training log will be saved to: {log_file}")
    except IOError as e:
        logging.error(f"Failed to create CSV log file {log_file}. Reason: {e}")
        CSV_WRITER = None

def log_to_csv(data_dict):
    """将单次迭代的数据写入CSV文件"""
    global CSV_WRITER, LOG_FILE_HANDLE
    if CSV_WRITER:
        CSV_WRITER.writerow(data_dict)
        if data_dict['Iteration'] % 100 == 0:
            LOG_FILE_HANDLE.flush()

def close_csv_logger():
    """关闭CSV日志文件句柄"""
    global LOG_FILE_HANDLE
    if LOG_FILE_HANDLE:
        LOG_FILE_HANDLE.close()
        logging.info("CSV training log file closed.")

# --- UDF 辅助函数 (来自您的原始脚本) ---
def get_dynamic_weight_map_udf(scene, gaussians, viewspace_points_tensor, visibility_filter, opt):
    """
    Calculates a dynamic weight map based on 3D UDF distances.
    """
    if not hasattr(scene, 'udf_grid_manager') or scene.udf_grid_manager is None or scene.udf_grid_manager.udf_grid is None:
        return None

    try:
        H = int(viewspace_points_tensor.shape[1])
        W = int(viewspace_points_tensor.shape[2])
        
        with torch.no_grad():
            udf_distances_all = scene.udf_grid_manager.query(gaussians.get_xyz)

        visible_udf_distances = udf_distances_all[visibility_filter].squeeze()
        visible_points_2d = viewspace_points_tensor[visibility_filter]

        proxy_map = torch.zeros((H, W), device="cuda")
        
        pixel_x = torch.round((visible_points_2d[:, 0].detach() + 1) * (W - 1) / 2).long()
        pixel_y = torch.round((visible_points_2d[:, 1].detach() + 1) * (H - 1) / 2).long()
        
        valid_mask = (pixel_x >= 0) & (pixel_x < W) & (pixel_y >= 0) & (pixel_y < H)
        
        if not valid_mask.any():
            return None

        flat_indices = pixel_y[valid_mask] * W + pixel_x[valid_mask]
        proxy_map.view(-1).scatter_add_(0, flat_indices, visible_udf_distances[valid_mask])
        
        if opt.udf_blur_radius > 0:
            kernel_size = opt.udf_blur_radius * 2 + 1
            proxy_map_smooth = torch.nn.functional.avg_pool2d(
                proxy_map.unsqueeze(0).unsqueeze(0), 
                kernel_size=kernel_size, stride=1, padding=opt.udf_blur_radius
            ).squeeze()
        else:
            proxy_map_smooth = proxy_map

        dynamic_weight_map = opt.udf_dynamic_lambda * proxy_map_smooth
        return dynamic_weight_map

    except Exception as e:
        logging.warning(f"Error during UDF dynamic weight map calculation: {e}. Skipping for this iteration.")
        return None

def training(dataset, opt, pipe, testing_iterations, saving_iterations, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    setup_csv_logger(dataset.model_path)
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, opt, gaussians)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    time_accumulators = { 
        "data_loading": 0.0, "render": 0.0, "loss_calc": 0.0, "backward": 0.0, 
        "optimizer_step": 0.0, "densification": 0.0, "total_iteration": 0.0 
    }
    last_report_iter = first_iter
    
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress")
    
    for iteration in progress_bar:
        torch.cuda.synchronize(); iter_total_start_time = time.time()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        torch.cuda.synchronize(); time_start = time.time()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image = viewpoint_cam.original_image.cuda()
        torch.cuda.synchronize(); time_accumulators["data_loading"] += time.time() - time_start
        
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        torch.cuda.synchronize(); time_start = time.time()
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        rendered_depth = render_pkg.get("rendered_depth")
        rendered_normals = render_pkg.get("rendered_normals")
        dynamic_error_map = render_pkg.get("dynamic_error_map")
        torch.cuda.synchronize(); time_accumulators["render"] += time.time() - time_start

        torch.cuda.synchronize(); time_start = time.time()
        
        # 初始化损失项
        total_loss = 0
        l1_loss_val = torch.tensor(0.0, device="cuda")
        ssim_loss_val = torch.tensor(0.0, device="cuda")
        line_loss_val = torch.tensor(0.0, device="cuda")
        normal_loss_val = torch.tensor(0.0, device="cuda")
        depth_loss_val = torch.tensor(0.0, device="cuda")
        
        pixel_error_l1 = torch.abs(image - gt_image)
        
        total_weight_map = torch.ones_like(gt_image[0:1, :, :])
        
        if opt.geometry_constraint_type == 'line' and iteration > opt.geometry_start_iter:
            static_map = viewpoint_cam.static_loss_weight_map
            if static_map is not None:
                total_weight_map = static_map
                
            if dynamic_error_map is not None:
                total_weight_map = total_weight_map + opt.line_dynamic_lambda * dynamic_error_map
            
            line_loss_val = ((total_weight_map - 1.0) * pixel_error_l1).mean()
        
        elif opt.geometry_constraint_type == 'udf' and iteration > opt.geometry_start_iter:
            udf_dynamic_map = get_dynamic_weight_map_udf(scene, gaussians, viewspace_point_tensor, visibility_filter, opt)
            if udf_dynamic_map is not None:
                total_weight_map = 1.0 + udf_dynamic_map.unsqueeze(0)
        
        l1_loss_val = (total_weight_map * pixel_error_l1).mean()
        ssim_loss_val = 1.0 - ssim(image, gt_image)
        render_loss = (1.0 - opt.lambda_dssim) * l1_loss_val + opt.lambda_dssim * ssim_loss_val
        total_loss += render_loss

        if opt.geometry_constraint_type in ['normal', 'normal_depth'] and iteration > opt.geometry_start_iter:
            gt_normals = viewpoint_cam.gt_normal_map
            if gt_normals is not None and rendered_normals is not None:
                foreground_mask = (rendered_depth > 0).detach()
                cos_sim = F.cosine_similarity(rendered_normals, gt_normals, dim=0)
                normal_loss = (1.0 - cos_sim)[foreground_mask.squeeze(0)]
                if normal_loss.numel() > 0: # Avoid taking mean of empty tensor
                    normal_loss_val = normal_loss.mean() * opt.lambda_normals
                    total_loss += normal_loss_val
        
        if opt.geometry_constraint_type in ['depth', 'normal_depth'] and iteration > opt.geometry_start_iter:
            gt_depth = viewpoint_cam.gt_depth_map
            if gt_depth is not None and rendered_depth is not None:
                foreground_mask = (gt_depth > 0).detach()
                if torch.any(foreground_mask): # Avoid loss on empty masks
                    depth_loss = F.l1_loss(rendered_depth[foreground_mask], gt_depth[foreground_mask])
                    depth_loss_val = depth_loss * opt.lambda_depth
                    total_loss += depth_loss_val
        
        torch.cuda.synchronize(); time_accumulators["loss_calc"] += time.time() - time_start
        
        torch.cuda.synchronize(); time_start = time.time()
        total_loss.backward()
        torch.cuda.synchronize(); time_accumulators["backward"] += time.time() - time_start
        
        torch.cuda.synchronize(); iter_total_end_time = time.time()
        time_accumulators["total_iteration"] += iter_total_end_time - iter_total_start_time

        with torch.no_grad():
            torch.cuda.synchronize(); time_start = time.time()
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            torch.cuda.synchronize(); time_accumulators["densification"] += time.time() - time_start

            torch.cuda.synchronize(); time_start = time.time()
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            torch.cuda.synchronize(); time_accumulators["optimizer_step"] += time.time() - time_start

            current_psnr = psnr(image, gt_image).mean().item()
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.{6}f}",
                "PSNR": f"{current_psnr:.2f}",
                "Gaussians": f"{gaussians.get_xyz.shape[0]}"
            })

            training_report(tb_writer, iteration, total_loss, l1_loss_val, ssim_loss_val, testing_iterations, scene, render, (pipe, background),
                            line_loss_val, normal_loss_val, depth_loss_val)
            
            log_data = {
                'Iteration': iteration,
                'Total_Loss': total_loss.item(),
                'L1_Loss': l1_loss_val.item(),
                'SSIM_Loss': ssim_loss_val.item(),
                'Line_Loss(part)': line_loss_val.item(),
                'Normal_Loss(part)': normal_loss_val.item(),
                'Depth_Loss(part)': depth_loss_val.item(),
                'Train_PSNR': current_psnr,
                'Total_Points': gaussians.get_xyz.shape[0],
                'Total_Iter_Time_s': iter_total_end_time - iter_total_start_time
            }
            log_to_csv(log_data)
            
            if (iteration in saving_iterations):
                logging.info(f"\n[ITERATION {iteration}] Saving model")
                scene.save(iteration)
            
            if iteration % 1000 == 0 and iteration > first_iter:
                num_iters_since_report = iteration - last_report_iter
                avg_times = {k: v / num_iters_since_report for k, v in time_accumulators.items()}
                logging.info(f"\n--- [ Timing Report - Iteration {iteration} ] ---")
                logging.info(f"Average time over last {num_iters_since_report} iterations (ms/iter):")
                total_reported_time_ms = sum(v for k, v in avg_times.items() if k != "total_iteration") * 1000
                logging.info(f"  - Data Loading:        {avg_times['data_loading']*1000:>8.2f} ms")
                logging.info(f"  - Rendering (Forward):   {avg_times['render']*1000:>8.2f} ms")
                logging.info(f"  - Loss Calculation:      {avg_times['loss_calc']*1000:>8.2f} ms")
                logging.info(f"  - Backward Pass:         {avg_times['backward']*1000:>8.2f} ms")
                logging.info(f"  - Optimizer Step:        {avg_times['optimizer_step']*1000:>8.2f} ms")
                logging.info(f"  - Densification:         {avg_times['densification']*1000:>8.2f} ms")
                logging.info(f"  -------------------------------------------")
                logging.info(f"  Sum of parts:            {total_reported_time_ms:>8.2f} ms")
                logging.info(f"  Total Iteration Time:    {avg_times['total_iteration']*1000:>8.2f} ms")
                logging.info(f"--- [ Report End ] ---\n")
                
                for key in time_accumulators: time_accumulators[key] = 0.0
                last_report_iter = iteration

    progress_bar.close()
    close_csv_logger()
    logging.info("\nTraining complete.")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    logging.info(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        logging.warning("Tensorboard not found: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, total_loss, l1_loss, ssim_loss, testing_iterations, scene, render, renderArgs, line_loss, normal_loss, depth_loss):
    if tb_writer:
        tb_writer.add_scalar('train_loss/total', total_loss, iteration)
        tb_writer.add_scalar('train_loss/l1_weighted', l1_loss, iteration)
        tb_writer.add_scalar('train_loss/ssim', ssim_loss, iteration)
        tb_writer.add_scalar('train_loss_components/line', line_loss, iteration)
        tb_writer.add_scalar('train_loss_components/normal', normal_loss, iteration)
        tb_writer.add_scalar('train_loss_components/depth', depth_loss, iteration)
        tb_writer.add_scalar('gaussians/count', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()})
        
        for config in [validation_configs]:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test, psnr_test = 0.0, 0.0
                for idx, viewpoint in enumerate(tqdm(config['cameras'], desc=f"Evaluating [{config['name']}]")):
                    render_pkg = render(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(f'{config["name"]}_view_{viewpoint.image_name}_{viewpoint.uid}/render', image[None], global_step=iteration)
                        # Optionally visualize depth and normals during testing
                        if render_pkg.get("rendered_depth") is not None:
                            depth_img = render_pkg["rendered_depth"]
                            # Normalize for visualization
                            depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
                            tb_writer.add_images(f'{config["name"]}_view_{viewpoint.image_name}_{viewpoint.uid}/depth', depth_img, global_step=iteration)
                        if render_pkg.get("rendered_normals") is not None:
                            norm_img = (render_pkg["rendered_normals"] / 2.0 + 0.5)
                            tb_writer.add_images(f'{config["name"]}_view_{viewpoint.image_name}_{viewpoint.uid}/normals', norm_img, global_step=iteration)
                        tb_writer.add_images(f'{config["name"]}_view_{viewpoint.image_name}_{viewpoint.uid}/ground_truth', gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logging.info(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.5f} PSNR {psnr_test:.2f}")
                if tb_writer:
                    tb_writer.add_scalar(f'val_metrics/{config["name"]}_psnr', psnr_test, iteration)
                    tb_writer.add_scalar(f'val_metrics/{config["name"]}_l1', l1_test, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 20000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 20000])
    parser.add_argument("--quiet", action='store_true')
    
    args = get_combined_args(parser)
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    safe_state(args.quiet)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.debug_from)

    print("\nTraining complete.")