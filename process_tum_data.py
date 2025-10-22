# -*- coding: utf-8 -*-
"""
使用 OpenCV 的 Line Segment Detector (LSD) 高效处理 TUM RGB-D 数据集以提取直线特征。
此脚本整合了所有修正：
1. 使用 OpenCV 替代深度学习模型。
2. 能够过滤掉指定长度以下的短线段。
3. 能够自动处理 TUM 数据集常见的嵌套目录结构。
"""
import os
import sys
import argparse
import tarfile
import cv2
import numpy as np
import json
from tqdm import tqdm

def process_dataset(dataset_path, num_visualizations, min_line_length=30):
    """
    处理单个 TUM RGB-D 数据集目录。
    - 提取 RGB 图像
    - 运行 OpenCV LSD 直线检测
    - 过滤掉过短的线段
    - 保存 lines.json
    - 保存可视化图像
    """
    print(f"\n正在处理数据集: {dataset_path}")
    print(f"将只保留长度大于 {min_line_length} 像素的线段。")

    rgb_dir = os.path.join(dataset_path, 'rgb')
    associations_file = os.path.join(dataset_path, 'associations.txt')
    
    if not os.path.exists(rgb_dir) or not os.path.exists(associations_file):
        print(f"警告: 在 {dataset_path} 中未找到 'rgb' 目录或 'associations.txt' 文件，已跳过。")
        return

    vis_dir = os.path.join(dataset_path, 'line_visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    lsd = cv2.createLineSegmentDetector(0)

    all_lines_data = {}
    image_paths = []
    
    with open(associations_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_paths.append(parts[1])

    vis_count = 0
    for image_path_suffix in tqdm(image_paths, desc="检测并过滤直线中"):
        image_filename = os.path.basename(image_path_suffix)
        full_image_path = os.path.join(dataset_path, image_path_suffix)
        
        if not os.path.exists(full_image_path):
            continue
        
        img = cv2.imread(full_image_path, cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        lines, _, _, _ = lsd.detect(gray_img)
        
        lines_for_json = []
        filtered_lines_for_vis = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if length > min_line_length:
                    lines_for_json.append([float(x1), float(y1), float(x2), float(y2), 1.0])
                    filtered_lines_for_vis.append(line)
        
        if vis_count < num_visualizations:
            drawn_img = img.copy()
            if filtered_lines_for_vis:
                drawn_img = lsd.drawSegments(drawn_img, np.array(filtered_lines_for_vis))
            
            vis_path = os.path.join(vis_dir, image_filename)
            cv2.imwrite(vis_path, drawn_img)
            vis_count += 1
        
        all_lines_data[image_filename] = lines_for_json

    json_path = os.path.join(dataset_path, 'lines.json')
    with open(json_path, 'w') as f:
        json.dump(all_lines_data, f, indent=4)
        
    print(f"成功保存 {json_path}")
    print(f"已保存 {vis_count} 张过滤后的可视化图像至 {vis_dir}")

def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_DATASET_ROOT = os.path.join(SCRIPT_DIR, "dataset")

    parser = argparse.ArgumentParser(description="使用 OpenCV 高效处理 TUM RGB-D 数据集以提取直线。")
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT,
                        help="包含 .tgz 数据集文件的目录路径。")
    parser.add_argument("--num_visualizations", type=int, default=5,
                        help="为每个数据集保存的带直线标注的示例图像数量。")
    parser.add_argument("--min_line_length", type=int, default=30,
                        help="过滤线段的最小像素长度，小于此长度的将被丢弃。")
    
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        print(f"错误: 数据集根目录 '{args.dataset_root}' 不存在。请检查路径。")
        sys.exit(1)

    for filename in sorted(os.listdir(args.dataset_root)):
        if filename.endswith(".tgz"):
            tgz_path = os.path.join(args.dataset_root, filename)
            dataset_name = filename.replace(".tgz", "")
            dataset_dir = os.path.join(args.dataset_root, dataset_name)

            if not os.path.exists(dataset_dir):
                print(f"正在解压 {tgz_path}...")
                try:
                    with tarfile.open(tgz_path, 'r:gz') as tar:
                        tar.extractall(path=args.dataset_root)
                    print("解压完成。")
                except tarfile.ReadError:
                    print(f"警告: 无法读取 {tgz_path}。文件可能已损坏或非有效 tgz 文件，已跳过。")
                    continue
            else:
                print(f"数据集目录 '{dataset_dir}' 已存在，跳过解压。")

            processing_path = dataset_dir
            if not os.path.exists(os.path.join(processing_path, 'rgb')):
                nested_dir_path = os.path.join(dataset_dir, dataset_name)
                if os.path.isdir(nested_dir_path):
                    print(f"检测到嵌套目录结构，将处理路径调整为: {nested_dir_path}")
                    processing_path = nested_dir_path

            process_dataset(processing_path, args.num_visualizations, args.min_line_length)

if __name__ == "__main__":
    main()