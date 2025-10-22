import cv2
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
import sys

# [核心修改 1/3] 将项目根目录添加到 Python 路径，以便导入 colmap_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text
except ImportError:
    print("❌ 错误: 无法从 'scene.colmap_loader' 导入函数。")
    print("   请确保您是从 gaussian-splatting 项目的根目录运行此脚本。")
    sys.exit(1)


# [核心新增] 新增函数，用于从 images.bin 文件中读取图像名
def read_colmap_image_names_from_bin(images_bin_path):
    """
    从 COLMAP 的 images.bin 文件中解析出所有图像的文件名。
    """
    extrinsics = read_extrinsics_binary(images_bin_path)
    # extrinsics 是一个字典，值为 CameraInfo 对象，其 .name 属性就是文件名
    image_names = [ext.name for ext in extrinsics.values()]
    return sorted(list(set(image_names)))

def read_colmap_image_names_from_txt(images_txt_path):
    """
    (原始函数) 从 COLMAP 的 images.txt 文件中解析出所有图像的文件名。
    """
    image_names = []
    with open(images_txt_path, "r") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("#"): continue
        # 在 images.txt 中，每两行描述一个图像，第二行包含文件名
        if (i - 4) % 2 == 0 and i >= 4:
            parts = line.split()
            if len(parts) > 9:
                image_names.append(parts[-1])
    return sorted(list(set(image_names)))

def detect_lines_in_scene(dataset_path, image_dir="images", visualize=False, 
                          detector='lsd', 
                          min_filter_length=40,
                          hough_threshold=150, min_length=50, max_gap=10):
    """
    对指定场景中的所有图像运行直线检测，并生成 lines.json 文件。
    现在可以自动检测并读取 images.bin 或 images.txt。
    """
    sparse_dir = os.path.join(dataset_path, "sparse", "0")
    images_bin_path = os.path.join(sparse_dir, "images.bin")
    images_txt_path = os.path.join(sparse_dir, "images.txt")
    image_folder = os.path.join(dataset_path, image_dir)
    image_names = []

    # [核心修改 2/3] 智能判断应该读取 .bin 还是 .txt 文件
    if os.path.exists(images_bin_path):
        print(f"✅ 找到二进制 COLMAP 文件: '{images_bin_path}'")
        try:
            image_names = read_colmap_image_names_from_bin(images_bin_path)
        except Exception as e:
            print(f"❌ 错误: 解析 '{images_bin_path}' 失败: {e}")
            return
    elif os.path.exists(images_txt_path):
        print(f"✅ 找到文本 COLMAP 文件: '{images_txt_path}'")
        try:
            image_names = read_colmap_image_names_from_txt(images_txt_path)
        except Exception as e:
            print(f"❌ 错误: 解析 '{images_txt_path}' 失败: {e}")
            return
    else:
        print(f"❌ 错误: 在 '{sparse_dir}' 目录下既未找到 'images.bin' 也未找到 'images.txt'。")
        return

    image_paths = [os.path.join(image_folder, name) for name in image_names]
    print(f"🔍 从 COLMAP 元数据中找到 {len(image_paths)} 张图像。使用 [{detector.upper()}] 检测器开始处理...")
    
    if detector == 'lsd':
        print(f"📏 将过滤掉所有长度小于 {min_filter_length} 像素的直线。")

    if visualize:
        vis_dir = os.path.join(dataset_path, "lines_visualization")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"🖼️ 可视化结果将保存至: '{vis_dir}'")

    line_detector = None
    if detector == 'lsd':
        line_detector = cv2.createLineSegmentDetector(0)
            
    all_lines_data = {}
    for image_path in tqdm(image_paths, desc=f"Processing {os.path.basename(dataset_path)}"):
        image_basename = os.path.basename(image_path)
        if not os.path.exists(image_path):
            all_lines_data[image_basename] = []
            continue

        try:
            image = cv2.imread(image_path)
            if image is None:
                all_lines_data[image_basename] = []
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            lines = None
            if detector == 'lsd':
                lines, _, _, _ = line_detector.detect(gray)
            elif detector == 'hough':
                lines = cv2.HoughLinesP(gray, 1, np.pi / 180, threshold=hough_threshold,
                                        minLineLength=min_length, maxLineGap=max_gap)
            
            filtered_lines = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if length >= min_filter_length:
                        filtered_lines.append([[int(x1), int(y1)], [int(x2), int(y2)]])
            
            all_lines_data[image_basename] = filtered_lines

            if visualize and filtered_lines:
                vis_image = image.copy()
                for line_coords in filtered_lines:
                    pt1 = tuple(line_coords[0])
                    pt2 = tuple(line_coords[1])
                    cv2.line(vis_image, pt1, pt2, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(vis_dir, image_basename), vis_image)

        except Exception as e:
            print(f"\n处理 '{image_path}' 时发生严重错误: {e}")
            all_lines_data[image_basename] = []

    output_path = os.path.join(dataset_path, "lines.json")
    try:
        with open(output_path, 'w') as f:
            json.dump(all_lines_data, f, indent=4)
        print(f"✅ 直线检测完成。结果已保存至: '{output_path}'")
    except Exception as e:
        print(f"❌ 错误: 无法保存json文件至 '{output_path}'. 原因: {e}")

if __name__ == "__main__":
    # [核心修改 3/3] 确保 argparse 的 description 更清晰
    parser = argparse.ArgumentParser(description="在 COLMAP 场景中检测直线，支持 .bin 和 .txt 格式。")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集根目录路径 (例如: .../nerf_360/bicycle)")
    parser.add_argument("--image_dir", type=str, default="images", help="存放图像的子目录名 (例如: images_4)")
    parser.add_argument("--detector", type=str, default='lsd', choices=['lsd', 'hough'], help="选择使用的直线检测算法")
    parser.add_argument("--min_filter_length", type=int, default=40, help="后处理过滤器：保留的最小直线长度")
    parser.add_argument("--visualize", action='store_true', help="是否生成并保存带有直线的可视化图像")
    
    # 霍夫变换专用参数
    parser.add_argument("--hough_threshold", type=int, default=150, help="[仅Hough] 霍夫变换的阈值")
    parser.add_argument("--min_length", type=int, default=50, help="[仅Hough] 霍夫变换检测的最小线段长度")
    parser.add_argument("--max_gap", type=int, default=10, help="[仅Hough] 霍夫变换允许的最大线段间隙")
    
    args = parser.parse_args()
    
    detect_lines_in_scene(args.dataset_path, args.image_dir, args.visualize,
                          args.detector, args.min_filter_length,
                          args.hough_threshold, args.min_length, args.max_gap)