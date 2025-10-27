import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def generate_priors_for_file(normal_map_path, output_curvature_path, output_confidence_path):
    """
    为单个法线图文件生成并保存曲率图和置信度图。
    """
    try:
        # 1. 读取并预处理法线图
        normal_img = cv2.imread(normal_map_path, cv2.IMREAD_COLOR)
        if normal_img is None:
            print(f"    - 警告: 无法读取图像文件，跳过: {os.path.basename(normal_map_path)}")
            return False
            
        # [0, 255] -> [-1, 1]
        normal_img_float = (normal_img.astype(np.float32) / 255.0) * 2.0 - 1.0

        # 2. 生成置信度图 (Confidence Map)
        magnitudes = np.linalg.norm(normal_img_float, axis=2)
        # 背景区域通常是(0,0,0)，模长为0。有效法线模长应接近1。
        confidence_map = (magnitudes > 0.1).astype(np.uint8) * 255
        cv2.imwrite(output_confidence_path, confidence_map)

        # 3. 生成曲率图 (Curvature Map)
        grad_x = cv2.Sobel(normal_img_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(normal_img_float, cv2.CV_32F, 0, 1, ksize=3)
        curvature = np.sqrt(np.sum(grad_x**2, axis=2) + np.sum(grad_y**2, axis=2))

        # 归一化到 [0, 255] 以便保存
        p98 = np.percentile(curvature, 98)
        if p98 > 1e-6:
            curvature = np.clip(curvature, 0, p98) / p98
        
        curvature_map_uint8 = (curvature * 255).astype(np.uint8)
        cv2.imwrite(output_curvature_path, curvature_map_uint8)
        
        return True

    except Exception as e:
        print(f"    - ❌ 处理文件时发生错误 {os.path.basename(normal_map_path)}: {e}")
        return False

def process_directory(normals_dir):
    """
    处理整个法线图目录。
    """
    print(f"=== 开始批量处理目录: {normals_dir} ===")
    
    # 检查输入目录是否存在
    if not os.path.isdir(normals_dir):
        print(f"❌ 错误: 输入的目录不存在: '{normals_dir}'")
        return

    # 1. 路径设置与目录创建
    base_dir = os.path.dirname(normals_dir)
    curvature_dir = os.path.join(base_dir, "gt_curvature")
    confidence_dir = os.path.join(base_dir, "gt_confidence")

    os.makedirs(curvature_dir, exist_ok=True)
    os.makedirs(confidence_dir, exist_ok=True)
    print(f"    -> 输出曲率图至: {curvature_dir}")
    print(f"    -> 输出置信度图至: {confidence_dir}")

    # 2. 查找所有法线图文件
    normal_files = [f for f in os.listdir(normals_dir) if f.endswith("_normal.png")]
    if not normal_files:
        print("    - 警告: 在目录中没有找到 `*_normal.png` 文件。")
        return
        
    print(f"    -> 找到了 {len(normal_files)} 个法线图文件。开始处理...")

    # 3. 循环处理每个文件
    success_count = 0
    fail_count = 0
    for filename in tqdm(normal_files, desc="处理法线图"):
        base_filename = filename.replace('_normal.png', '')
        
        input_path = os.path.join(normals_dir, filename)
        output_curvature_path = os.path.join(curvature_dir, f"{base_filename}_curvature.png")
        output_confidence_path = os.path.join(confidence_dir, f"{base_filename}_confidence.png")
        
        if generate_priors_for_file(input_path, output_curvature_path, output_confidence_path):
            success_count += 1
        else:
            fail_count += 1
            
    print(f"--- 批量处理完成！ ---")
    print(f"    -> ✅ 成功处理: {success_count} 个文件")
    if fail_count > 0:
        print(f"    -> ❌ 失败: {fail_count} 个文件")
    print("========================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从一个法线图目录批量生成几何先验图（曲率和置信度）。")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="包含所有 `*_normal.png` 文件的 `gt_normals` 目录的路径。")
    
    args = parser.parse_args()
    
    process_directory(args.input_dir)