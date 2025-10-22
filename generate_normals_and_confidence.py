# =================================================================================
# =================================================================================
#               从3DGS模型生成高质量几何图 (深度/法线/信度/曲率)
#
#   版本：V9 - 全自动批处理终极版
#   作者：Gemini AI
#
#   本脚本是一个全自动、多场景的解决方案，具备以下特性：
#   - 批处理：自动扫描并处理 Mip-NeRF 360 数据集下的所有场景。
#   - 智能解压：自动查找并解压每个场景对应的3DGS模型压缩包。
#   - 高质量几何：集成三阶段深度图优化流程（去噪、插值填充、双边滤波平滑）。
#   - GPU加速：核心计算使用PyTorch在CUDA上完成。
#   - 鲁棒性：自动处理路径、分辨率和常见噪声问题。
#
#   使用方法:
#   1. 确认下方 [ 1. 配置区域 ] 中的两个根目录路径与您的环境完全匹配。
#   2. 确保已安装必要的库: pip install torch numpy opencv-python plyfile tqdm scipy
#   3. 在终端中直接运行: python generate_geometry_batch.py
#
# =================================================================================
# =================================================================================

import os
import numpy as np
import cv2
from plyfile import PlyData
from tqdm import tqdm
import torch
import torch.nn.functional as F
import zipfile
from scipy.interpolate import griddata

# =================================================================================
# >>> [ 1. 配置区域 ] <<<
# 您只需要配置下面这两个根目录！
# =================================================================================

# 包含所有Mip-NeRF 360场景（如 bicycle, garden...）的根目录
NERF_360_ROOT_PATH = "/root/autodl-tmp/gaussian-splatting/data/nerf_360"

# 存放所有场景3DGS结果压缩包（如 kitchen.zip, garden.zip...）的目录
# 根据您的描述，压缩包位于 /data 目录下
ZIPPED_MODELS_PATH = "/root/autodl-tmp/gaussian-splatting/data"

# =================================================================================
# >>> [ 2. 核心功能函数 ] <<<
# =================================================================================
def load_ply_points(ply_path, device='cuda'):
    try:
        plydata = PlyData.read(ply_path)
        points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        return torch.from_numpy(points).to(device).float()
    except FileNotFoundError:
        print(f"错误: 找不到PLY文件: {ply_path}")
        return None
    except Exception as e:
        print(f"读取PLY文件时发生错误: {e}")
        return None

def depth_to_normals(depth, fx, fy, cx, cy):
    device = depth.device
    h, w = depth.shape
    v_grid, u_grid = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    z = depth
    x = (u_grid - cx) * z / fx
    y = (v_grid - cy) * z / fy
    pts = torch.stack((x, y, z), dim=0).unsqueeze(0)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).expand(3, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).expand(3, 1, 3, 3)
    tangent_u = F.conv2d(pts, sobel_x, padding=1, groups=3)
    tangent_v = F.conv2d(pts, sobel_y, padding=1, groups=3)
    normals = torch.cross(tangent_u.squeeze(0).permute(1, 2, 0), tangent_v.squeeze(0).permute(1, 2, 0), dim=2)
    norm = torch.linalg.norm(normals, dim=-1, keepdim=True)
    norm[norm < 1e-6] = 1e-6
    normals = normals / norm
    normals[normals[..., 2] > 0] *= -1.0
    return normals.permute(2, 0, 1)

def confidence_from_depth_gradient(depth_tensor):
    depth_np = depth_tensor.cpu().numpy()
    if depth_np.max() > 0:
        max_grad_clip = np.percentile(depth_np[depth_np > 0], 99) / 10.0
    else:
        max_grad_clip = 0.1
    sobel_x = cv2.Sobel(depth_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_np, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_magnitude = np.clip(grad_magnitude, 0, max_grad_clip)
    max_val = grad_magnitude.max()
    uncertainty = grad_magnitude / max_val if max_val > 0 else np.zeros_like(grad_magnitude)
    confidence_map = 1.0 - uncertainty
    confidence_map[depth_np <= 0] = 0.0
    return torch.from_numpy(confidence_map).float()

def curvature_from_normals(normals_tensor):
    normals_np = normals_tensor.permute(1, 2, 0).cpu().numpy()
    laplacian_x = cv2.Laplacian(normals_np[:,:,0], cv2.CV_32F, ksize=3)
    laplacian_y = cv2.Laplacian(normals_np[:,:,1], cv2.CV_32F, ksize=3)
    laplacian_z = cv2.Laplacian(normals_np[:,:,2], cv2.CV_32F, ksize=3)
    curvature = np.sqrt(laplacian_x**2 + laplacian_y**2 + laplacian_z**2)
    if curvature.max() > 0:
        max_curv = np.percentile(curvature, 99.5)
        if max_curv > 0:
            curvature = np.clip(curvature / max_curv, 0, 1)
    return torch.from_numpy(curvature).float()

def refine_depth_map(depth_raw_tensor, median_ksize=3, interpolation_method='cubic', bilateral_d=5, bilateral_sigma=1.0):
    """(⭐ 核心改进) 通过三阶段流程优化原始深度图：去噪、填充孔洞、平滑。"""
    depth_np = depth_raw_tensor.cpu().numpy().astype(np.float32)
    h, w = depth_np.shape
    
    # 阶段一: 初始中值滤波去噪
    depth_denoised = cv2.medianBlur(depth_np, median_ksize)
    
    # 阶段二: 高质量孔洞填充
    valid_mask = depth_denoised > 1e-6
    if not np.any(valid_mask): return torch.from_numpy(depth_denoised)
    
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    known_points_coords = np.vstack([xx[valid_mask], yy[valid_mask]]).T
    known_depth_values = depth_denoised[valid_mask]
    
    filled_depth = griddata(known_points_coords, known_depth_values, (xx, yy), method=interpolation_method, fill_value=0)
    
    # 阶段三: 最终双边滤波平滑
    if filled_depth.max() > 0:
        norm_depth = filled_depth / filled_depth.max()
        norm_smoothed = cv2.bilateralFilter(norm_depth.astype(np.float32), bilateral_d, bilateral_sigma, bilateral_sigma)
        final_depth_np = norm_smoothed * filled_depth.max()
    else:
        final_depth_np = filled_depth
        
    return torch.from_numpy(final_depth_np.astype(np.float32))

# =================================================================================
# >>> [ 3. 主执行流程 ] <<<
# =================================================================================

if __name__ == "__main__":
    print("初始化并检查环境...")
    if not torch.cuda.is_available():
        print("错误: 本脚本需要CUDA支持的GPU才能运行。")
        exit()
    device = torch.device("cuda")
    print(f"使用设备: {torch.cuda.get_device_name(0)}")

    # 获取所有场景目录
    try:
        scene_names = [d for d in os.listdir(NERF_360_ROOT_PATH) if os.path.isdir(os.path.join(NERF_360_ROOT_PATH, d))]
    except FileNotFoundError:
        print(f"错误: 找不到Mip-NeRF 360根目录: {NERF_360_ROOT_PATH}")
        print("请检查 NERF_360_ROOT_PATH 配置是否正确。")
        exit()

    print(f"发现 {len(scene_names)} 个场景: {scene_names}")

    # --- 遍历所有场景 ---
    for scene_name in scene_names:
        print(f"\n{'='*25} 开始处理场景: {scene_name.upper()} {'='*25}")
        
        # --- 动态定义路径 ---
        scene_path = os.path.join(NERF_360_ROOT_PATH, scene_name)
        model_results_path = scene_path  # 模型解压后就放在场景目录里
        rgb_image_path = os.path.join(scene_path, "images_4")
        output_base_path = os.path.join(scene_path, "derived_data")

        # --- 智能解压 ---
        model_zip_path = os.path.join(ZIPPED_MODELS_PATH, f"{scene_name}.zip")
        # 检查一个关键子目录是否存在，来判断是否已解压
        if not os.path.isdir(os.path.join(model_results_path, "checkpoint")):
            if os.path.exists(model_zip_path):
                print(f"未找到解压后的模型，正在从 {model_zip_path} 解压...")
                with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(model_results_path)
                print("解压完成。")
            else:
                print(f"警告: 场景 '{scene_name}' 缺少对应的模型压缩包 {model_zip_path}，跳过此场景。")
                continue
        else:
            print("检测到已解压的模型，跳过解压步骤。")
            
        if not os.path.isdir(rgb_image_path):
            print(f"警告: 在场景 '{scene_name}' 中找不到RGB图像文件夹: {rgb_image_path}，跳过此场景。")
            continue

        # --- 加载3DGS模型 (每个场景加载一次) ---
        ply_file_path = os.path.join(model_results_path, "checkpoint/point_cloud/iteration_30000/point_cloud.ply")
        print(f"正在从 {ply_file_path} 加载高斯点云到GPU...")
        points_world = load_ply_points(ply_file_path, device=device)
        if points_world is None:
            print(f"错误: 加载点云失败，跳过场景 '{scene_name}'。")
            continue
            
        ones = torch.ones((points_world.shape[0], 1), device=device)
        points_world_homo = torch.cat([points_world, ones], dim=1)
        print(f"成功加载 {points_world.shape[0]} 个点。")

        # --- 准备输出目录 ---
        os.makedirs(output_base_path, exist_ok=True)
        depth_out_dir = os.path.join(output_base_path, "depth")
        normal_out_dir = os.path.join(output_base_path, "normal")
        confidence_out_dir = os.path.join(output_base_path, "confidence")
        curvature_out_dir = os.path.join(output_base_path, "curvature")
        os.makedirs(depth_out_dir, exist_ok=True)
        os.makedirs(normal_out_dir, exist_ok=True)
        os.makedirs(confidence_out_dir, exist_ok=True)
        os.makedirs(curvature_out_dir, exist_ok=True)
    
        # --- 遍历所有相机 ---
        cameras_path = os.path.join(model_results_path, "predictions/cameras")
        if not os.path.isdir(cameras_path):
            print(f"警告: 找不到相机参数文件夹: {cameras_path}，跳过场景 '{scene_name}'。")
            continue
        camera_files = sorted([f for f in os.listdir(cameras_path) if f.endswith('.npz')])
        
        print(f"找到 {len(camera_files)} 个相机视角，开始处理...")
        for cam_file in tqdm(camera_files, desc=f"处理 {scene_name} 视角"):
            cam_name = os.path.splitext(cam_file)[0]
            
            # 步骤 1: 动态读取图像分辨率
            possible_extensions = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']
            rgb_file_path = next((os.path.join(rgb_image_path, f"{cam_name}{ext}") for ext in possible_extensions if os.path.exists(os.path.join(rgb_image_path, f"{cam_name}{ext}"))), None)
            if rgb_file_path is None:
                tqdm.write(f"警告: 在 '{rgb_image_path}' 中找不到RGB图像 {cam_name}，跳过。")
                continue
            h, w, _ = cv2.imread(rgb_file_path).shape
            
            # 步骤 2: 获取相机参数
            cam_data = np.load(os.path.join(cameras_path, cam_file))
            try:
                intrinsics_data = torch.from_numpy(cam_data['intrinsics']).to(device).float()
                c2w_3x4 = torch.from_numpy(cam_data['poses']).to(device).float()
            except KeyError as e:
                tqdm.write(f"错误: 相机文件 {cam_file} 缺少Key: {e}，跳过。")
                continue

            fx, fy, cx, cy = intrinsics_data[0], intrinsics_data[1], intrinsics_data[2], intrinsics_data[3]
            c2w_4x4 = torch.cat([c2w_3x4, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device)], dim=0)
            w2c = torch.inverse(c2w_4x4)
            
            # 步骤 3: 投影点云生成原始深度图
            points_camera = (w2c @ points_world_homo.T).T
            points_camera_xyz = points_camera[:, :3]
            depths = points_camera_xyz[:, 2]
            u = (points_camera_xyz[:, 0] * fx / depths) + cx
            v = (points_camera_xyz[:, 1] * fy / depths) + cy
            mask = (depths > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u_idx, v_idx = u[mask].long(), v[mask].long()
            flat_indices = v_idx * w + u_idx
            flat_depth = torch.full((h * w,), float('inf'), device=device)
            flat_depth.scatter_reduce_(0, flat_indices, depths[mask], reduce='amin', include_self=False)
            depth_map_raw = flat_depth.view(h, w)
            depth_map_raw[depth_map_raw == float('inf')] = 0
            
            # 步骤 4: (⭐核心改进⭐) 使用高级三阶段流程优化深度图
            depth_map = refine_depth_map(depth_map_raw, interpolation_method='cubic').to(device)
            
            # 步骤 5: 计算所有衍生几何图
            with torch.no_grad():
                normals = depth_to_normals(depth_map, fx, fy, cx, cy)
                confidence = confidence_from_depth_gradient(depth_map)
                curvature = curvature_from_normals(normals)
            
            # 步骤 6: 保存结果
            cv2.imwrite(os.path.join(depth_out_dir, f"{cam_name}.png"), (depth_map.cpu().numpy() * 1000).astype(np.uint16))
            cv2.imwrite(os.path.join(normal_out_dir, f"{cam_name}.png"), cv2.cvtColor(((normals.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(confidence_out_dir, f"{cam_name}.png"), (confidence.numpy() * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(curvature_out_dir, f"{cam_name}.png"), (curvature.numpy() * 255).astype(np.uint8))

        print(f"[✔] 场景 {scene_name.upper()} 处理完毕！结果保存在: {output_base_path}")

    print("\n🎉🎉🎉 所有场景均已处理完毕！🎉🎉🎉")