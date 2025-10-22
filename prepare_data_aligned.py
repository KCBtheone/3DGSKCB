# 文件名: prepare_data_aligned.py (v2.3 - 终极融合版)
import os
import json
import shutil
import numpy as np
import argparse
import math
import sys
from tqdm import tqdm
from PIL import Image

def rotmat2qvec(R):
    """
    将旋转矩阵 (Rotation Matrix) 转换为四元数 (Quaternion Vector)。(与可用版本一致)
    """
    q_w = np.sqrt(max(0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if q_w < 1e-6:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            t = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q_x, q_y, q_z, q_w = 0.5 * t, (R[0, 1] + R[1, 0]) / (2.0 * t), (R[0, 2] + R[2, 0]) / (2.0 * t), (R[2, 1] - R[1, 2]) / (2.0 * t)
        elif R[1, 1] > R[2, 2]:
            t = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2])
            q_x, q_y, q_z, q_w = (R[0, 1] + R[1, 0]) / (2.0 * t), 0.5 * t, (R[1, 2] + R[2, 1]) / (2.0 * t), (R[0, 2] - R[2, 0]) / (2.0 * t)
        else:
            t = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2])
            q_x, q_y, q_z, q_w = (R[0, 2] + R[2, 0]) / (2.0 * t), (R[1, 2] + R[2, 1]) / (2.0 * t), 0.5 * t, (R[1, 0] - R[0, 1]) / (2.0 * t)
        return np.array([q_w, q_x, q_y, q_z])
    q_x = (R[2, 1] - R[1, 2]) / (4.0 * q_w)
    q_y = (R[0, 2] - R[2, 0]) / (4.0 * q_w)
    q_z = (R[1, 0] - R[0, 1]) / (4.0 * q_w)
    return np.array([q_w, q_x, q_y, q_z])

def process_scene(input_dir, output_dir, reference_transform=None):
    """
    处理单个场景，将其从 NeRF 的 transforms.json 格式转换为 3DGS 的 COLMAP 格式。
    """
    print(f"\n--- 正在处理场景: {os.path.basename(input_dir)} ---")
    
    train_transforms_path = os.path.join(input_dir, 'transforms_train.json')
    if not os.path.exists(train_transforms_path):
        train_transforms_path = os.path.join(input_dir, 'transforms.json')
    if not os.path.exists(train_transforms_path):
        print(f"❌ 错误: 在'{input_dir}'中找不到 'transforms_train.json' 或 'transforms.json'。")
        return None
        
    images_dir = os.path.join(output_dir, "images")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(images_dir); os.makedirs(sparse_dir)

    with open(train_transforms_path, 'r') as f:
        transforms_data = json.load(f)

    # ############################ 关键修正点 ############################
    #
    #   使用与您提供的“可用版本”完全相同的坐标系转换方法。
    #   这是最可能解决问题的核心步骤。
    #
    # ####################################################################
    print("应用 OpenCV -> OpenGL 坐标系转换...")
    all_c2w_matrices_opencv = [np.array(frame["transform_matrix"]) for frame in transforms_data['frames']]
    all_c2w_matrices_opengl = []
    for c2w in all_c2w_matrices_opencv:
        c2w_gl = c2w.copy()
        c2w_gl[:, 1:3] *= -1 # 翻转Y, Z轴
        all_c2w_matrices_opengl.append(c2w_gl)

    if reference_transform is None:
        print("  > 这是参考场景，正在计算新的归一化变换...")
        all_positions_opengl = np.array([m[:3, 3] for m in all_c2w_matrices_opengl])
        scene_center = np.mean(all_positions_opengl, axis=0)
        positions_recentered = all_positions_opengl - scene_center
        scene_scale = np.max(np.linalg.norm(positions_recentered, axis=1))
        if scene_scale < 1e-6: scene_scale = 1.0
        current_transform = {'center': scene_center.tolist(), 'scale': scene_scale}
    else:
        print("  > 正在使用参考场景的归一化变换...")
        scene_center = np.array(reference_transform['center'])
        scene_scale = reference_transform['scale']
        current_transform = reference_transform

    image_folder_name = "images" if os.path.exists(os.path.join(input_dir, "images")) else "rgb"
    rgb_folder = os.path.join(input_dir, image_folder_name)
    
    actual_image_files = sorted([f for f in os.listdir(rgb_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if len(actual_image_files) != len(transforms_data['frames']):
        sys.exit(f"❌ 错误: 图片数量 ({len(actual_image_files)}) 与相机位姿数量 ({len(transforms_data['frames'])}) 不匹配!")
        
    with Image.open(os.path.join(rgb_folder, actual_image_files[0])) as img: W, H = img.size
    fov_x = float(transforms_data["camera_angle_x"])
    fl_x = 0.5 * W / math.tan(0.5 * fov_x)
    with open(os.path.join(sparse_dir, "cameras.txt"), 'w') as f:
        f.write(f"1 PINHOLE {W} {H} {fl_x:.6f} {fl_x:.6f} {W/2.0:.6f} {H/2.0:.6f}\n")
    
    images_txt_lines = []
    for i, img_filename in enumerate(tqdm(actual_image_files, desc="处理帧")):
        shutil.copyfile(os.path.join(rgb_folder, img_filename), os.path.join(images_dir, img_filename))
        
        c2w_matrix = all_c2w_matrices_opengl[i]
        
        c2w_matrix[:3, 3] = (c2w_matrix[:3, 3] - scene_center) / scene_scale
        
        w2c_matrix = np.linalg.inv(c2w_matrix)
        R, t = w2c_matrix[:3, :3], w2c_matrix[:3, 3]
        qvec = rotmat2qvec(R)
        
        line1 = f"{i + 1} {qvec[0]:.8f} {qvec[1]:.8f} {qvec[2]:.8f} {qvec[3]:.8f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} 1 {img_filename}\n"
        images_txt_lines.extend([line1, "\n"]) # 保持正确的两行格式
        
    with open(os.path.join(sparse_dir, "images.txt"), 'w') as f: f.writelines(images_txt_lines)
    
    # 保持生成随机点云的逻辑
    num_pts = 10000
    xyz = (np.random.random((num_pts, 3)) * 2 - 1) * 1.5
    rgb = (np.random.randint(0, 256, (num_pts, 3))).astype(np.uint8)
    with open(os.path.join(sparse_dir, "points3D.txt"), 'w') as f:
        f.write("# 3D point list\n")
        for i in range(num_pts):
            p, c = xyz[i], rgb[i]
            f.write(f"{i+1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0.1 \n")
        
    print(f"✅ 场景处理完成！输出至: {os.path.abspath(output_dir)}")
    return current_transform

def main():
    parser = argparse.ArgumentParser(description="将多个NeRF数据集转换为对齐的3DGS格式。")
    parser.add_argument("--ref_dir", type=str, required=True, help="参考场景(Scene A)的原始数据目录")
    parser.add_argument("--other_dirs", type=str, nargs='+', required=True, help="其他需要对齐的场景(Scene B, C...)的原始数据目录")
    parser.add_argument("--output_base", type=str, required=True, help="所有处理后数据的根输出目录")
    args = parser.parse_args()

    ref_output_dir = os.path.join(args.output_base, os.path.basename(args.ref_dir))
    reference_transform = process_scene(args.ref_dir, ref_output_dir)

    if reference_transform is None:
        sys.exit("参考场景处理失败，流程终止。")

    for other_dir in args.other_dirs:
        other_output_dir = os.path.join(args.output_base, os.path.basename(other_dir))
        process_scene(other_dir, other_output_dir, reference_transform)

if __name__ == "__main__":
    main()