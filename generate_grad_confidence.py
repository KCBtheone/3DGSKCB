#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
#                      IMPORTS & SETUP
# ==============================================================================
import os
import sys
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation
import concurrent.futures
import traceback
import multiprocessing

try:
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.io import read_image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ==============================================================================
#           1. COLMAP 读取器 (保持不变)
# ==============================================================================
class ColmapCamera:
    def __init__(self, id, model, width, height, params):
        self.id, self.model, self.width, self.height, self.params = id, model, width, height, params

class ColmapImage:
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id, self.qvec, self.tvec, self.camera_id, self.name, self.xys, self.point3D_ids = id, qvec, tvec, camera_id, name, xys, point3D_ids
    def qvec2rotmat(self):
        return Rotation.from_quat([self.qvec[1], self.qvec[2], self.qvec[3], self.qvec[0]]).as_matrix()

def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"): continue
            elems = line.split()
            camera_id, model, width, height = int(elems[0]), elems[1], int(elems[2]), int(elems[3])
            params = np.array(tuple(map(float, elems[4:])))
            cameras[camera_id] = ColmapCamera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras

def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        i = 0
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"): continue
            i += 1
            if i % 2 == 1:
                elems = line.split()
                image_id, qvec = int(elems[0]), np.array(tuple(map(float, elems[1:5])))
                tvec, camera_id, image_name = np.array(tuple(map(float, elems[5:8]))), int(elems[8]), elems[9]
            else:
                elems = line.split()
                xys = np.array([float(e) for e in elems[0::3]])
                ys = np.array([float(e) for e in elems[1::3]])
                point3D_ids = np.array([int(e) for e in elems[2::3]])
                xys = np.stack([xys, ys], axis=1)
                images[image_id] = ColmapImage(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images

def get_camera_intrinsics_and_pose(camera: ColmapCamera, image: ColmapImage):
    params = camera.params
    if camera.model in ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"]:
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    else:
        fx, fy, cx, cy = params[0], params[0], camera.width / 2, camera.height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R = image.qvec2rotmat()
    t = image.tvec.reshape(3, 1)
    return K, R, t

# ==============================================================================
#           2. GPU 加速版核心函数 (最终OOM修复版 V2)
# ==============================================================================
NUM_SCALES = 3
GRADIENT_THRESHOLD = 15
NEIGHBOR_VIEWS_COUNT = 5
FINAL_BLUR_KERNEL_SIZE = 35
GAMMA_CORRECTION_VALUE = 0.2
EPIPOLAR_DISTANCE_THRESHOLD = 1.5
EPIPOLINE_BATCH_SIZE = 4096
# 新增: 对邻居像素也进行批处理，彻底解决OOM问题
NEIGHBOR_PIXEL_BATCH_SIZE = 65536 

def get_gradient_pyramid_torch(img_tensor, num_scales):
    scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32, device=img_tensor.device).view(1, 1, 3, 3)
    scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32, device=img_tensor.device).view(1, 1, 3, 3)
    pyramid = []
    current_img = img_tensor.unsqueeze(0)
    for _ in range(num_scales):
        grad_x = torch.nn.functional.conv2d(current_img, scharr_x, padding=1)
        grad_y = torch.nn.functional.conv2d(current_img, scharr_y, padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2).squeeze(0)
        pyramid.append(grad_mag)
        if len(pyramid) < num_scales:
            current_img = torch.nn.functional.avg_pool2d(current_img, kernel_size=2)
    return pyramid

def compute_gradient_consistency_for_image_gpu(args):
    target_image_obj, scene_path, all_images, all_cameras, device_id = args
    try:
        device = torch.device(f"cuda:{device_id}")
        with torch.no_grad():
            target_cam = all_cameras[target_image_obj.camera_id]
            h, w = target_cam.height, target_cam.width
            
            target_img_path = scene_path / "images" / target_image_obj.name
            if not target_img_path.exists():
                target_img_path = scene_path / target_image_obj.name
                if not target_img_path.exists(): return None, target_image_obj.name, "Image file not found"
            
            target_tensor = read_image(str(target_img_path)).to(device).float()
            target_gray = TF.rgb_to_grayscale(target_tensor)
            target_grad_pyramid = get_gradient_pyramid_torch(target_gray, NUM_SCALES)

            target_pids = set(target_image_obj.point3D_ids); target_pids.discard(-1)
            scores = sorted(
                [(len(set(obj.point3D_ids).intersection(target_pids)), obj) for obj in all_images.values() if obj.id != target_image_obj.id],
                key=lambda x: x[0], reverse=True
            )
            neighbors = [s[1] for s in scores[:NEIGHBOR_VIEWS_COUNT]]

            K_target, R_target, t_target = get_camera_intrinsics_and_pose(target_cam, target_image_obj)
            K_target_t = torch.from_numpy(K_target).float().to(device)
            R_target_t = torch.from_numpy(R_target).float().to(device)
            t_target_t = torch.from_numpy(t_target).float().to(device)
            K_target_inv_t = torch.inverse(K_target_t)
            final_score_map = torch.zeros((h, w), dtype=torch.float32, device=device)

            for neighbor_image_obj in neighbors:
                neighbor_img_path = scene_path / "images" / neighbor_image_obj.name
                if not neighbor_img_path.exists():
                    neighbor_img_path = scene_path / neighbor_image_obj.name
                    if not neighbor_img_path.exists(): continue
                
                neighbor_tensor = read_image(str(neighbor_img_path)).to(device).float()
                neighbor_gray = TF.rgb_to_grayscale(neighbor_tensor)
                neighbor_grad_pyramid = get_gradient_pyramid_torch(neighbor_gray, NUM_SCALES)

                neighbor_cam = all_cameras[neighbor_image_obj.camera_id]
                K_neighbor, R_neighbor, t_neighbor = get_camera_intrinsics_and_pose(neighbor_cam, neighbor_image_obj)
                K_neighbor_t = torch.from_numpy(K_neighbor).float().to(device)
                R_neighbor_t = torch.from_numpy(R_neighbor).float().to(device)
                t_neighbor_t = torch.from_numpy(t_neighbor).float().to(device)
                K_neighbor_inv_t = torch.inverse(K_neighbor_t)
                R_rel, t_rel = R_neighbor_t @ R_target_t.T, t_neighbor_t - (R_neighbor_t @ R_target_t.T) @ t_target_t
                t_rel_x = torch.tensor([
                    [0, -t_rel[2,0], t_rel[1,0]], [t_rel[2,0], 0, -t_rel[0,0]], [-t_rel[1,0], t_rel[0,0], 0]
                ], device=device)
                F = K_neighbor_inv_t.T @ t_rel_x @ R_rel @ K_target_inv_t

                for l in range(NUM_SCALES):
                    scale = 2**l
                    target_grad_mag = target_grad_pyramid[l].squeeze(0)
                    neighbor_grad_mag = neighbor_grad_pyramid[l].squeeze(0)
                    threshold = GRADIENT_THRESHOLD / scale

                    strong_pixels_coords_target = torch.nonzero(target_grad_mag > threshold, as_tuple=False)
                    num_strong_pixels = strong_pixels_coords_target.shape[0]
                    if num_strong_pixels == 0: continue

                    strong_pixels_coords_neighbor = torch.nonzero(neighbor_grad_mag > threshold, as_tuple=False)
                    num_strong_pixels_neighbor = strong_pixels_coords_neighbor.shape[0]
                    if num_strong_pixels_neighbor == 0: continue
                    
                    strong_neighbor_homo = torch.cat([strong_pixels_coords_neighbor[:, [1, 0]].float(), torch.ones(num_strong_pixels_neighbor, 1, device=device)], dim=1)
                    
                    pts_target_homo = torch.cat([strong_pixels_coords_target[:, [1, 0]].float(), torch.ones(num_strong_pixels, 1, device=device)], dim=1)
                    epilines_all = F @ pts_target_homo.T
                    
                    has_match_total = torch.zeros(num_strong_pixels, dtype=torch.bool, device=device)

                    for i in range(0, num_strong_pixels, EPIPOLINE_BATCH_SIZE):
                        batch_start = i
                        batch_end = min(i + EPIPOLINE_BATCH_SIZE, num_strong_pixels)
                        epilines_batch = epilines_all[:, batch_start:batch_end]
                        current_target_batch_size = epilines_batch.shape[1]
                        
                        # 初始化当前目标批次的匹配状态
                        has_match_batch = torch.zeros(current_target_batch_size, dtype=torch.bool, device=device)

                        # --- [ 🚀 CRITICAL FIX V2: 对邻居像素进行批处理 🚀 ] ---
                        distances_denominator = torch.sqrt(epilines_batch[0,:]**2 + epilines_batch[1,:]**2)
                        
                        for j in range(0, num_strong_pixels_neighbor, NEIGHBOR_PIXEL_BATCH_SIZE):
                            neighbor_batch_start = j
                            neighbor_batch_end = min(j + NEIGHBOR_PIXEL_BATCH_SIZE, num_strong_pixels_neighbor)
                            strong_neighbor_homo_batch = strong_neighbor_homo[neighbor_batch_start:neighbor_batch_end]

                            # 核心矩阵乘法现在变得非常小: (NEIGHBOR_PIXEL_BATCH_SIZE x EPIPOLINE_BATCH_SIZE)
                            distances_numerator = torch.abs(strong_neighbor_homo_batch @ epilines_batch)
                            distances = distances_numerator / distances_denominator
                            
                            is_on_line_chunk = distances < EPIPOLAR_DISTANCE_THRESHOLD
                            
                            # 检查在当前邻居块中，哪些目标点找到了匹配
                            match_in_chunk = torch.any(is_on_line_chunk, dim=0)
                            
                            # 使用逻辑"或"累积匹配结果
                            has_match_batch = has_match_batch | match_in_chunk
                            
                            # 优化: 如果这个批次的所有目标点都已经找到了匹配，就没必要再检查剩下的邻居点了
                            if torch.all(has_match_batch):
                                break
                        # --- [ FIX END ] ---

                        has_match_total[batch_start:batch_end] = has_match_batch

                    if not torch.any(has_match_total): continue
                    
                    matched_coords = strong_pixels_coords_target[has_match_total]
                    matched_grads = target_grad_mag[matched_coords[:, 0], matched_coords[:, 1]]
                    orig_y = matched_coords[:, 0] * scale
                    orig_x = matched_coords[:, 1] * scale
                    
                    for y_add in range(scale):
                        for x_add in range(scale):
                            y_offset, x_offset = orig_y + y_add, orig_x + x_add
                            mask = (y_offset < h) & (x_offset < w)
                            if not torch.any(mask): continue
                            indices = (y_offset[mask] * w + x_offset[mask]).long()
                            final_score_map.view(-1).index_add_(0, indices, matched_grads[mask])

        return final_score_map.cpu().numpy(), target_image_obj.name, None
    except Exception as e:
        tb_str = traceback.format_exc()
        return None, target_image_obj.name, f"Error in GPU processing for {target_image_obj.name}: {e}\n{tb_str}"

# ==============================================================================
#                      3. 主控函数 (保持不变)
# ==============================================================================
def main():
    parser = ArgumentParser(description="从COLMAP稀疏重建并行生成高质量的几何信度图 (GPU加速最终版-OOM修复V2)。")
    # ------------------ 可配置参数 ------------------
    # 在下方列表中添加或删除您想要处理的场景名称
    SCENES_TO_PROCESS = [
        "kicker", "courtyard", "delivery_area", "electro", "facade",
        "meadow", "office", "pipes", "playground", "relief", "relief_2"
        # 例如: "my_custom_scene_1", "my_custom_scene_2"
    ]
    # 根据您的GPU显存和CPU核心数调整。对于32GB显存的GPU，2或3是比较安全的值。
    # 如果您有多个GPU，可以适当增加这个值。
    NUM_WORKERS = 2
    
    # 修改为您的项目根目录
    # 示例: PROJECT_DIR = Path("/path/to/your/gaussian-splatting")
    PROJECT_DIR = Path("/root/autodl-tmp/gaussian-splatting")
    
    # 通常不需要修改这个
    DATA_ROOT_DIR = PROJECT_DIR / "data"
    # ---------------------------------------------------

    if not (TORCH_AVAILABLE and torch.cuda.is_available()):
        print("❌ Critical Error: PyTorch或CUDA未找到。此脚本需要GPU才能运行。正在中止。")
        return

    print("✅ PyTorch 和 CUDA 已找到。将使用GPU进行加速。")
    print(f"   将使用 {NUM_WORKERS} 个工作进程。如果遇到CUDA内存问题，请尝试减少此值。")

    for scene_name in SCENES_TO_PROCESS:
        scene_path = DATA_ROOT_DIR / scene_name
        if not scene_path.is_dir():
            print(f"❌ 错误: 场景路径 '{scene_path}' 不存在。正在跳过。")
            continue

        # 自动检测 sparse/0 或 sparse 目录
        sparse_dir = scene_path / "sparse/0"
        if not sparse_dir.is_dir():
            sparse_dir = scene_path / "sparse"
            if not sparse_dir.is_dir():
                 print(f"❌ 错误: 未能在 '{scene_path}' 下找到稀疏重建目录 ('sparse/0' 或 'sparse')。正在跳过。")
                 continue

        output_dir = scene_path / "geometry_priors"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"\n--- 正在读取场景 '{scene_name}' 的COLMAP数据 ---")
        try:
            cameras = read_cameras_text(sparse_dir / "cameras.txt")
            images = read_images_text(sparse_dir / "images.txt")
        except Exception as e:
            print(f"❌ 错误: 加载 '{scene_name}' 的COLMAP文件时出错: {e}")
            continue

        num_gpus = torch.cuda.device_count()
        print(f"   发现 {num_gpus} 个可用的GPU。任务将被分配到这些GPU上。")
        tasks = [(img, scene_path, images, cameras, i % num_gpus) for i, img in enumerate(images.values())]
        
        print(f"\n--- 正在为 {len(tasks)} 张图像计算梯度一致性 ({NUM_WORKERS} 个工作进程) ---")
        
        final_confidence_maps = {}
        # 使用 'spawn' 上下文以避免CUDA在fork中初始化的问题
        mp_context = multiprocessing.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=mp_context) as executor:
            results = list(tqdm(executor.map(compute_gradient_consistency_for_image_gpu, tasks), total=len(tasks), desc=f"处理中 {scene_name}"))
            
            error_count = 0
            for score_map, image_name, err in results:
                if err:
                    # 只打印前几次错误详情，避免刷屏
                    if error_count < 5:
                        print(f"  -> 错误: 图像 {image_name} 处理失败。详情:\n{err}")
                    error_count += 1
                    continue
                if score_map is None or np.all(score_map == 0):
                    continue

                # 后处理步骤
                score_max = np.max(score_map)
                normalized_map = score_map / score_max if score_max > 0 else score_map
                blurred_map = cv2.GaussianBlur(normalized_map, (FINAL_BLUR_KERNEL_SIZE, FINAL_BLUR_KERNEL_SIZE), 0)
                gamma_corrected_map = np.power(blurred_map, GAMMA_CORRECTION_VALUE)
                final_confidence_maps[image_name] = gamma_corrected_map
            
            if error_count > 5:
                print(f"  -> ... (另有 {error_count - 5} 个错误未显示)")

        print(f"\n--- 正在保存 {len(final_confidence_maps)} 张新的置信度图 ---")
        for image_name, final_map in tqdm(final_confidence_maps.items(), desc="保存中"):
            output_map = (final_map * 255.0).astype(np.uint8)
            stem = Path(image_name).stem
            output_path = output_dir / f"{stem}_grad_conf_gamma.png"
            cv2.imwrite(str(output_path), output_map)

        print(f"\n✅ 场景 '{scene_name}' 的所有新置信度图已保存至 '{output_dir}'")
    
    print("\n🎉 所有场景处理完毕。")

if __name__ == "__main__":
    # 确保在使用 'spawn' 或 'forkserver' 时，主逻辑被这个if块包裹
    main()