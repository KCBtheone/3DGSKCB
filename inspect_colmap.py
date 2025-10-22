




#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================================================
#                      IMPORTS & SETUP
# ==============================================================================
import sys
import numpy as np
import struct
from argparse import ArgumentParser
from pathlib import Path
from scipy.spatial.transform import Rotation

# ==============================================================================
#           COLMAP 读取器 (从您的脚本中提取)
# ==============================================================================
CAMERA_MODELS = {0: ('SIMPLE_PINHOLE', 3), 1: ('PINHOLE', 4), 2: ('SIMPLE_RADIAL', 4), 3: ('RADIAL', 5), 4: ('OPENCV', 8), 5: ('OPENCV_FISHEYE', 8), 6: ('FULL_OPENCV', 12), 7: ('FOV', 5), 8: ('SIMPLE_RADIAL_FISHEYE', 4), 9: ('RADIAL_FISHEYE', 5), 10: ('THIN_PRISM_FISHEYE', 12)}
class ColmapCamera:
    def __init__(self, id, model, width, height, params): self.id, self.model, self.width, self.height, self.params = id, model, width, height, params
class ColmapImage:
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids): self.id, self.qvec, self.tvec, self.camera_id, self.name, self.xys, self.point3D_ids = id, qvec, tvec, camera_id, name, xys, point3D_ids
class ColmapPoint3D:
    def __init__(self, id, xyz, error): self.id, self.xyz, self.error = id, xyz, error

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = struct.unpack('<Q', fid.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack('<iIII', fid.read(16))
            model_name, num_params = CAMERA_MODELS[model_id]
            params = struct.unpack('<' + 'd' * num_params, fid.read(8 * num_params))
            cameras[camera_id] = ColmapCamera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = struct.unpack('<Q', fid.read(8))[0]
        for _ in range(num_reg_images):
            image_id, = struct.unpack('<i', fid.read(4))
            qvec = struct.unpack('<4d', fid.read(32))
            tvec = struct.unpack('<3d', fid.read(24))
            camera_id, = struct.unpack('<i', fid.read(4))
            image_name = "".join(iter(lambda: fid.read(1).decode("utf-8"), "\0"))
            num_points2D = struct.unpack('<Q', fid.read(8))[0]
            if num_points2D > 0:
                data = struct.unpack('<' + 'ddq' * num_points2D, fid.read(24 * num_points2D))
                xys = np.array(data[0::3] + data[1::3]).reshape(2, num_points2D).T
                point3D_ids = np.array(data[2::3], dtype=np.int64)
            else:
                xys = np.empty((0, 2))
                point3D_ids = np.empty((0,), dtype=np.int64)
            images[image_id] = ColmapImage(id=image_id, qvec=np.array(qvec), tvec=np.array(tvec), camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images

def read_points3D_text(path):
    points3D = {}
    try:
        with open(path, "r") as fid:
            for line in fid:
                line = line.strip()
                if len(line) == 0 or line.startswith("#"): continue
                elems = line.split()
                point_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                error = float(elems[7])
                points3D[point_id] = ColmapPoint3D(id=point_id, xyz=xyz, error=error)
    except FileNotFoundError:
        return None # Return None if file doesn't exist
    return points3D

# ==============================================================================
#                      主控函数
# ==============================================================================
def main():
    parser = ArgumentParser(description="检查并打印 COLMAP 稀疏模型文件的内容。")
    parser.add_argument("colmap_model_path", type=str, help="COLMAP 稀疏模型目录的路径 (例如, 'data/facade/sparse/0')。")
    args = parser.parse_args()
    
    model_path = Path(args.colmap_model_path)
    
    if not model_path.exists():
        print(f"❌ 错误: 路径不存在 -> {model_path}")
        sys.exit(1)

    # --- 1. 检查 Cameras ---
    cameras_bin_path = model_path / "cameras.bin"
    print(f"============================================================")
    print(f"   1. 正在读取相机信息: {cameras_bin_path}")
    print(f"------------------------------------------------------------")
    if not cameras_bin_path.exists():
        print(f"❌ 文件未找到!")
    else:
        try:
            cameras = read_cameras_binary(cameras_bin_path)
            print(f"✅ 成功读取 {len(cameras)} 个相机模型。\n")
            for cam_id, cam in sorted(cameras.items()):
                print(f"  [相机 ID: {cam.id}]")
                print(f"    - 模型: {cam.model}")
                print(f"    - 宽度 (Width):  {cam.width}")
                print(f"    - 高度 (Height): {cam.height}  <--- ⚠️ 请重点检查此值是否为 0")
                print(f"    - 参数 (Params): {cam.params}")
                print("-" * 20)
        except Exception as e:
            print(f"❌ 读取 cameras.bin 时出错: {e}")

    # --- 2. 检查 Images ---
    images_bin_path = model_path / "images.bin"
    print(f"\n============================================================")
    print(f"   2. 正在读取图像信息: {images_bin_path}")
    print(f"------------------------------------------------------------")
    if not images_bin_path.exists():
        print(f"❌ 文件未找到!")
    else:
        try:
            images = read_images_binary(images_bin_path)
            print(f"✅ 成功读取 {len(images)} 张注册图像的信息。\n")
            # 只打印前5张作为样本
            for i, (img_id, img) in enumerate(sorted(images.items())):
                if i >= 5:
                    print("  ... (只显示前5张图像信息) ...")
                    break
                print(f"  [图像 ID: {img.id}]")
                print(f"    - 文件名: {img.name}")
                print(f"    - 相机 ID: {img.camera_id} <--- 确认这是否与上面有问题的相机ID对应")
                print(f"    - 2D特征点数量: {len(img.xys)}")
                print("-" * 20)
        except Exception as e:
            print(f"❌ 读取 images.bin 时出错: {e}")
            
    # --- 3. 检查 Points3D ---
    points_txt_path = model_path / "points3D.txt"
    print(f"\n============================================================")
    print(f"   3. 正在读取 3D 点云信息: {points_txt_path}")
    print(f"------------------------------------------------------------")
    if not points_txt_path.exists():
        print(f"❌ 文件未找到!")
    else:
        try:
            points3D = read_points3D_text(points_txt_path)
            if points3D is not None:
                print(f"✅ 成功读取 {len(points3D)} 个 3D 点。")
            else:
                 print(f"❌ 文件未找到!")
        except Exception as e:
            print(f"❌ 读取 points3D.txt 时出错: {e}")

    print(f"\n============================================================")
    print("检查完毕。")


if __name__ == "__main__":
    main()