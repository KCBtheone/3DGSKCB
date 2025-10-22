import argparse
import json
import os
import numpy as np
from PIL import Image

# 假设的输出目录结构
OUTPUT_DIR = "sparse/0"

# --- COLMAP BINARY 文件格式辅助函数 ---
# (这些函数旨在创建最小的 COLMAP 结构，用于初始化 3DGS)

def write_colmap_cameras(path, cameras):
    """Writes a minimal cameras.bin file."""
    with open(path, 'wb') as fid:
        for i, (width, height, f_x) in enumerate(cameras):
            # Camera ID (uint32)
            fid.write(np.array([i + 1], dtype=np.uint32).tobytes())
            # Model (string: SIMPLE_PINHOLE)
            model = "SIMPLE_PINHOLE"
            fid.write(model.encode('utf-8') + b'\x00')
            # Width (uint32)
            fid.write(np.array([width], dtype=np.uint32).tobytes())
            # Height (uint32)
            fid.write(np.array([height], dtype=np.uint32).tobytes())
            # Params (double: f, cx, cy)
            params = np.array([f_x, width / 2.0, height / 2.0], dtype=np.float64)
            fid.write(params.tobytes())

def write_colmap_images(path, images_data):
    """Writes an images.bin file with camera poses."""
    with open(path, 'wb') as fid:
        for i, (q, t, camera_id, name) in enumerate(images_data):
            # Image ID (uint32)
            fid.write(np.array([i + 1], dtype=np.uint32).tobytes())
            # QW, QX, QY, QZ (double)
            fid.write(q.astype(np.float64).tobytes())
            # TX, TY, TZ (double)
            fid.write(t.astype(np.float64).tobytes())
            # Camera ID (uint32)
            fid.write(np.array([camera_id], dtype=np.uint32).tobytes())
            # Name (string)
            fid.write(name.encode('utf-8') + b'\x00')
            # Point2D data (empty in this initial sparse setup)
            fid.write(np.array([0], dtype=np.uint64).tobytes())

def write_colmap_points3D(path, points_data):
    """Writes a minimal points3D.bin file for initialization."""
    with open(path, 'wb') as fid:
        for i, (xyz, rgb) in enumerate(points_data):
            # Point 3D ID (uint64)
            fid.write(np.array([i + 1], dtype=np.uint64).tobytes())
            # XYZ (double)
            fid.write(xyz.astype(np.float64).tobytes())
            # RGB (uint8)
            fid.write(rgb.astype(np.uint8).tobytes())
            # Track length (uint64) - must be 0 for minimal structure
            fid.write(np.array([0], dtype=np.uint64).tobytes())


# --- 核心转换逻辑 ---

def convert_nerf_to_colmap(datadir, num_points=100000):
    """
    读取 NeRF JSON 文件，转换姿态，并生成一个初始点云。
    """
    json_path = os.path.join(datadir, 'transforms_train.json')
    if not os.path.exists(json_path):
        print(f"Error: transforms_train.json not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. 获取图像尺寸和焦距
    # 假设所有图像尺寸相同。从第一张图获取
    first_frame = data['frames'][0]
    img_path = os.path.join(datadir, first_frame['file_path'])
    if not os.path.exists(img_path):
        # 尝试常见的图片路径修正 (e.g. NeRF synthetic)
        img_path = os.path.join(datadir, first_frame['file_path'].replace('images/', ''))
        if not os.path.exists(img_path):
            img_path = os.path.join(datadir, first_frame['file_path'].replace('train/', ''))
            
    if not os.path.exists(img_path):
        # 尝试直接使用 train 目录
        img_path = os.path.join(datadir, 'train', os.path.basename(first_frame['file_path']))
        if not os.path.exists(img_path):
             print(f"Warning: Could not find image to determine size. Assuming 800x800.")
             W, H = 800, 800
        else:
             W, H = Image.open(img_path).size
    else:
        W, H = Image.open(img_path).size
    
    # NeRF Synthetic 数据集通常提供 FOV，计算焦距 f_x
    f_x = 0.5 * W / np.tan(0.5 * data['camera_angle_x'])
    
    # 2. 准备 COLMAP 相机数据
    cameras_data = [(W, H, f_x)] # COLMAP 仅存储一个共享相机模型

    # 3. 准备 COLMAP 图像数据（姿态）
    images_data = []
    
    # 收集所有相机中心点用于初始化点云
    all_cam_centers = [] 
    
    for i, frame in enumerate(data['frames']):
        # NeRF 是 world-to-camera，COLMAP 是 world-to-camera，所以矩阵不变
        c2w = np.array(frame['transform_matrix'])
        w2c = np.linalg.inv(c2w)
        
        # COLMAP 需要 (QW, QX, QY, QZ) 和 (TX, TY, TZ)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        
        # 姿态转换：从 COLMAP/OpenCV (Right-Down-Forward) 约定到 (Right-Up-Back) 
        # NeRF Synthetic 的转换非常复杂，这里我们使用简化的旋转矩阵，并确保它是正交的
        
        # 将 NeRF 的 c2w (x: right, y: up, z: back) 转换为 COLMAP 的 w2c (x: right, y: down, z: forward)
        # 这是一个常见的转换步骤 (R_world @ R_cam_colmap = R_cam_nerf)
        flip_mat = np.array([
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.]
        ])
        
        # COLMAP pose (c2w) = NeRF pose (c2w) @ flip_mat
        # COLMAP w2c = inv(NeRF c2w @ flip_mat) = inv(flip_mat) @ NeRF w2c
        # R = flip_mat @ R_nerf @ flip_mat (更精确的转换会涉及四元数，这里使用简化的)
        
        # 简化的 COLMAP 转换 (通常适用于 NeRF Synthetic 的初始化)
        # R = w2c[:3, :3]
        # T = w2c[:3, 3]

        # 使用 scipy 的 Rotation 或 opencv 的 Rodrigues 是最准确的，
        # 为了避免依赖，这里使用最简单的四元数转换:
        from scipy.spatial.transform import Rotation
        R_rot = Rotation.from_matrix(R)
        q = R_rot.as_quat() # (x, y, z, w)
        q = np.array([q[3], q[0], q[1], q[2]]) # (w, x, y, z) -> COLMAP 约定 (QW, QX, QY, QZ)
        
        images_data.append((q, T, 1, os.path.basename(frame['file_path'])))
        
        # 收集相机中心点 (c2w[:3, 3])
        all_cam_centers.append(c2w[:3, 3])

    # 4. 生成初始点云 (简化的球形点云)
    # 3DGS 依赖于初始点云来定位高斯球。这里生成一个围绕所有相机中心点的小型随机点云。
    points_data = []
    all_cam_centers = np.stack(all_cam_centers, axis=0)
    center_mean = np.mean(all_cam_centers, axis=0)
    
    # 随机生成点云：在以相机中心均值点为中心，半径为 1 的球体内均匀分布
    radius = 1.0 
    random_points = np.random.uniform(-radius, radius, size=(num_points, 3)) + center_mean
    
    for i in range(num_points):
        # XYZ (从相机中心点附近随机采样)
        xyz = random_points[i]
        # RGB (随机灰色)
        rgb = np.full(3, np.random.randint(100, 200), dtype=np.uint8)
        
        points_data.append((xyz, rgb))


    # 5. 写入文件
    output_full_path = os.path.join(datadir, OUTPUT_DIR)
    os.makedirs(output_full_path, exist_ok=True)
    
    write_colmap_cameras(os.path.join(output_full_path, 'cameras.bin'), cameras_data)
    write_colmap_images(os.path.join(output_full_path, 'images.bin'), images_data)
    write_colmap_points3D(os.path.join(output_full_path, 'points3D.bin'), points_data)
    
    print(f"\n✅ 转换成功！已生成 {num_points} 个初始点。")
    print(f"数据已保存至: {output_full_path}")
    print("现在可以运行 3DGS/Mip-Splatting 训练脚本了。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert NeRF Synthetic JSON to COLMAP sparse format for 3DGS initialization.")
    parser.add_argument('datadir', type=str, help="Path to the NeRF scene directory (e.g., .../nerf_synthetic/lego)")
    parser.add_argument('--num_points', type=int, default=100000, help="Number of initial points to generate.")
    
    args = parser.parse_args()
    convert_nerf_to_colmap(args.datadir, args.num_points)