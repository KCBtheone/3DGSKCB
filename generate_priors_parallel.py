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

class ColmapPoint3D:
    def __init__(self, id, xyz, error):
        self.id, self.xyz, self.error = id, xyz, error

def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model, width, height = elems[1], int(elems[2]), int(elems[3])
            params = np.array(tuple(map(float, elems[4:])))
            cameras[camera_id] = ColmapCamera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras

def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        i = 0
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            i += 1
            if i % 2 == 1:
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
            else:
                elems = line.split()
                xys = np.array([float(e) for e in elems[0::3]])
                ys = np.array([float(e) for e in elems[1::3]])
                point3D_ids = np.array([int(e) for e in elems[2::3]])
                xys = np.stack([xys, ys], axis=1)
                images[image_id] = ColmapImage(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images

def read_points3D_text(path):
    points3D = {}
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            elems = line.split()
            point_id = int(elems[0])
            xyz = np.array(tuple(map(float, elems[1:4])))
            error = float(elems[7])
            points3D[point_id] = ColmapPoint3D(id=point_id, xyz=xyz, error=error)
    return points3D

# ==============================================================================
#                      2. 几何先验生成函数 (保持不变)
# ==============================================================================
def get_camera_intrinsics(camera: ColmapCamera):
    params = camera.params
    if camera.model in ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"]:
        fx = params[0]
        fy = params[1] if camera.model in ["PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"] else fx
        cx = params[2] if camera.model != "SIMPLE_PINHOLE" else camera.width / 2
        cy = params[3] if camera.model in ["PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"] else camera.height / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        fx, fy, cx, cy = params[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def project_points_to_maps(image: ColmapImage, camera: ColmapCamera, points3D: dict):
    R, t = image.qvec2rotmat(), image.tvec
    h, w = camera.height, camera.width
    depth_map = np.zeros((h, w), dtype=np.float32)
    error_map = np.zeros((h, w), dtype=np.float32)
    valid_mask = image.point3D_ids != -1
    valid_pids, valid_xys = image.point3D_ids[valid_mask], image.xys[valid_mask]
    world_points, errors = [], []
    for pid in valid_pids:
        if pid in points3D:
            world_points.append(points3D[pid].xyz)
            errors.append(points3D[pid].error)
    if not world_points: return depth_map, error_map
    world_points, errors = np.array(world_points), np.array(errors)
    camera_points = (R @ world_points.T).T + t
    depths = camera_points[:, 2]
    u, v = valid_xys[:, 0].astype(int), valid_xys[:, 1].astype(int)
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depths > 0)
    depth_map[v[mask], u[mask]] = depths[mask]
    error_map[v[mask], u[mask]] = errors[mask]
    return depth_map, error_map

def create_confidence_map(sparse_error_map, h, w, max_error_threshold=2.0, blur_kernel_size=25):
    if sparse_error_map is None or sparse_error_map.size == 0 or np.all(sparse_error_map == 0):
        return np.zeros((h, w), dtype=np.float32)
    normalized_error = np.clip(sparse_error_map / max_error_threshold, 0, 1)
    confidence = np.zeros_like(normalized_error)
    mask = sparse_error_map > 0
    confidence[mask] = 1.0 - normalized_error[mask]
    inpainted_confidence = cv2.inpaint(confidence, (1-mask).astype(np.uint8), 5, cv2.INPAINT_NS)
    if inpainted_confidence is None or inpainted_confidence.size == 0:
        return np.zeros((h, w), dtype=np.float32)
    blurred_confidence = cv2.GaussianBlur(inpainted_confidence, (blur_kernel_size, blur_kernel_size), 0)
    return blurred_confidence

def depth_to_normals(depth_map, K, h, w):
    if depth_map is None or np.all(depth_map == 0):
        return np.zeros((h, w, 3), dtype=np.float32)
    fx, fy = K[0, 0], K[1, 1]
    zx = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
    normal = np.dstack((-zx * fx, -zy * fy, np.ones_like(depth_map)))
    n = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= np.maximum(n, 1e-6)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[depth_map > 0] = normal[depth_map > 0]
    return normals

# ==============================================================================
#           3. 单张图像处理函数 (为并行化而封装)
# ==============================================================================
def process_single_image(args):
    """
    处理单张图像，生成并保存深度图、法线图和置信度图。
    设计为可被多进程调用。
    """
    image, camera, points3D, output_dir = args
    try:
        stem = Path(image.name).stem
        h, w = camera.height, camera.width
        
        sparse_depth, sparse_error = project_points_to_maps(image, camera, points3D)
        
        confidence_map = create_confidence_map(sparse_error, h, w)
        confidence_to_save = (confidence_map * 255.0).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{stem}_confidence.png"), confidence_to_save)

        mask = (sparse_depth > 0).astype(np.uint8)
        inpainted_depth = np.zeros_like(sparse_depth)
        if np.any(mask):
            inpainted_depth = cv2.inpaint(sparse_depth, 1 - mask, 5, cv2.INPAINT_NS)
        depth_to_save = (inpainted_depth * 1000.0).astype(np.uint16)
        cv2.imwrite(str(output_dir / f"{stem}.png"), depth_to_save)

        K = get_camera_intrinsics(camera)
        normals = depth_to_normals(inpainted_depth, K, h, w)
        normal_to_save = ((normals + 1.0) / 2.0 * 255.0).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"{stem}_normal.png"), cv2.cvtColor(normal_to_save, cv2.COLOR_RGB2BGR))

        return None
    except Exception as e:
        return f"Error processing {image.name}: {e}"

# ==============================================================================
#                      4. 主控函数 (多核并行)
# ==============================================================================
def process_scene(scene_path, num_workers):
    """
    为单个场景目录加载数据并并行生成几何先验。
    """
    scene_path = Path(scene_path)
    sparse_dir = scene_path / "sparse/0"
    output_dir = scene_path / "geometry_priors"
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n--- [Generator] Reading COLMAP TXT files for scene: {scene_path.name} ---")
    try:
        cameras_txt_path = sparse_dir / "cameras.txt"
        images_txt_path = sparse_dir / "images.txt"
        points3D_txt_path = sparse_dir / "points3D.txt"

        if not all([p.exists() for p in [cameras_txt_path, images_txt_path, points3D_txt_path]]):
            raise FileNotFoundError(f"Error: Missing required TXT files. Ensure cameras.txt, images.txt, and points3D.txt exist in '{sparse_dir}'.")

        cameras = read_cameras_text(cameras_txt_path)
        images = read_images_text(images_txt_path)
        points3D = read_points3D_text(points3D_txt_path)
        
        print(f"✅ [Generator] COLMAP TXT files for '{scene_path.name}' loaded successfully.")
    except Exception as e:
        print(f"❌ [Generator] Error loading COLMAP files for '{scene_path.name}': {e}")
        return

    tasks = []
    for image in images.values():
        if image.camera_id not in cameras:
            print(f"WARNING: Camera ID {image.camera_id} for image {image.name} not in cameras.txt, skipping.")
            continue
        
        camera = cameras[image.camera_id]
        if camera.height == 0 or camera.width == 0:
            print(f"CRITICAL ERROR: Invalid dimensions ({camera.width}x{camera.height}) for camera {camera.id}. Skipping image {image.name}.")
            continue
            
        tasks.append((image, camera, points3D, output_dir))
    
    if not tasks:
        print(f"🤷 [Generator] No valid images found to process for scene '{scene_path.name}'.")
        return

    print(f"\n--- [Generator] Generating priors for {len(tasks)} images in '{scene_path.name}' using {num_workers} workers ---")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks), desc=f"Processing {scene_path.name}"))
        
        for result in results:
            if result is not None:
                print(f"ERROR_DETAIL: {result}")

    print(f"\n✅ [Generator] All priors for scene '{scene_path.name}' saved to '{output_dir}'")

def main():
    parser = ArgumentParser(description="从COLMAP稀疏重建并行生成几何先验（深度、法线和置信度）。")
    parser.add_argument("scene_paths", nargs='+', type=str, help="一个或多个场景根目录的路径。")
    parser.add_argument("--num_workers", type=int, default=8, help="用于并行处理的CPU核心数。")
    args = parser.parse_args()
    
    for scene_path_str in args.scene_paths:
        scene_path = Path(scene_path_str)
        # ==================== [ ⭐ BUG修复 ⭐ ] ====================
        #  将 isdir() 修改为 is_dir()
        if not scene_path.is_dir():
        # =========================================================
            print(f"❌ Error: Provided path '{scene_path_str}' is not a valid directory. Skipping.")
            continue
        process_scene(scene_path, args.num_workers)

    print("\n🎉 All scenes processed.")

if __name__ == "__main__":
    main()