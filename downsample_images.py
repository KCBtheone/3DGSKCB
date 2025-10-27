import os
import argparse
from PIL import Image
from tqdm import tqdm
import sys

# 增加 Pillow 图像大小限制，防止因图片过大而被拒绝处理
Image.MAX_IMAGE_PIXELS = None 

def find_image_directory(base_path):
    """
    Finds the actual directory containing image files.
    It first checks the base_path itself, then its immediate subdirectories.
    """
    # 检查基本路径本身
    if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(base_path)):
        return base_path

    # 检查子目录
    for sub_dir in os.listdir(base_path):
        sub_dir_path = os.path.join(base_path, sub_dir)
        if os.path.isdir(sub_dir_path):
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(sub_dir_path)):
                print(f"Detected images are in a subdirectory: '{sub_dir_path}'")
                return sub_dir_path
    
    return base_path # 如果都找不到，返回原始路径让后续代码报错

def downsample_images(source_dir, factor):
    """
    Downsamples all images by a given factor. It automatically finds the image
    subdirectory (e.g., 'images/dslr_images_undistorted').
    """
    base_input_path = os.path.join(source_dir, "images")
    output_path = os.path.join(source_dir, f"images_{factor}")

    if not os.path.exists(base_input_path):
        print(f"Error: Base input directory '{base_input_path}' not found.")
        sys.exit(1)

    # 自动查找包含图片的真实目录
    actual_input_path = find_image_directory(base_input_path)

    if os.path.exists(output_path):
        print(f"Warning: Output directory '{output_path}' already exists. Files might be overwritten.")
    else:
        os.makedirs(output_path)
        print(f"Created output directory: '{output_path}'")

    try:
        image_files = [f for f in os.listdir(actual_input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    except FileNotFoundError:
        print(f"Error: Could not find image directory at '{actual_input_path}' or its subdirectories.")
        sys.exit(1)


    if not image_files:
        print(f"Error: No images found in '{actual_input_path}'. Please check the directory.")
        sys.exit(1)

    print(f"Found {len(image_files)} images to downsample by a factor of {factor}.")

    for filename in tqdm(image_files, desc="Downsampling images"):
        try:
            with Image.open(os.path.join(actual_input_path, filename)) as img:
                original_size = img.size
                new_size = (original_size[0] // factor, original_size[1] // factor)
                
                # 使用 LANCZOS 进行高质量降采样
                resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # 直接将降采样后的图片保存在 images_4 根目录下
                resized_img.save(os.path.join(output_path, filename))
        except Exception as e:
            print(f"\nError processing {filename}: {e}. This might be a memory issue for very large images.")
            print("Skipping this file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced image downsampling script for 3DGS datasets.")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="Path to the scene directory (e.g., 'data/office').")
    parser.add_argument("-r", "--resolution_factor", type=int, required=True, help="The factor by which to downsample the images (e.g., 2, 4, 8).")
    
    args = parser.parse_args()
    
    downsample_images(args.source_path, args.resolution_factor)
    print("\nDownsampling complete!")