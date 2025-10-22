# 文件名: convert_colmap.py (v2.2 - 变量名 Bug 修复版)
# 描述: 修复了 v2.1 版本中因笔误导致的 `UnboundLocalError`。
#       将 `t_c2w = -R_c2w @ t_c2w` 修正为 `t_c2w = -R_c2w @ t_w2c`。

import os
import sys
import json
import numpy as np
import argparse
from tqdm import tqdm
from scipy.spatial.transform import Rotation

def convert_txt_to_3dgs(source_path):
    sparse_txt_path = os.path.join(source_path, "sparse_txt")
    if not os.path.exists(sparse_txt_path):
        sys.exit(f"❌ 错误: 文本模型目录 '{sparse_txt_path}' 不存在。请确保 run.sh 脚本已成功将其生成。")

    cam_intrinsics_path = os.path.join(sparse_txt_path, "cameras.txt")
    cam_extrinsics_path = os.path.join(sparse_txt_path, "images.txt")

    # 1. 读取内参 (cameras.txt)
    cam_intrinsics = {}
    with open(cam_intrinsics_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            cam_id, model, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            params = np.array([float(p) for p in parts[4:]])
            cam_intrinsics[cam_id] = {'model': model, 'width': width, 'height': height, 'params': params}
            
    # 2. 读取外参 (images.txt) 并转换
    json_cameras = []
    with open(cam_extrinsics_path, 'r') as f:
        i = 0
        for line in tqdm(f, desc="  > 正在处理相机姿态"):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            i += 1
            if i % 2 != 1:
                continue

            parts = line.split()
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id, img_name = int(parts[8]), parts[9]
            
            intr = cam_intrinsics[cam_id]
            width, height, params = intr['width'], intr['height'], intr['params']

            R_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t_w2c = np.array([tx, ty, tz])
            R_c2w = R_w2c.T
            
            # [核心修复] 使用正确的变量名 t_w2c
            t_c2w = -R_c2w @ t_w2c

            coord_transform = np.diag([1, -1, -1])
            R_final = R_c2w @ coord_transform
            t_final = t_c2w

            if intr['model'] in ["SIMPLE_RADIAL", "SIMPLE_PINHOLE"]: fx = fy = params[0]
            elif intr['model'] == "PINHOLE": fx = fy = params[0], params[1]
            else: fx = fy = params[0]
            
            json_camera = {"id": len(json_cameras), "img_name": img_name, "width": width, "height": height, "position": t_final.tolist(), "rotation": R_final.tolist(), "fy": fy, "fx": fx}
            json_cameras.append(json_camera)

    json_cameras.sort(key=lambda x: x['img_name'])
    output_json_path = os.path.join(source_path, "cameras.json")
    with open(output_json_path, 'w') as f:
        json.dump(json_cameras, f, indent=4)
    print(f"  > ✅ `cameras.json` 已成功生成。")

def main():
    parser = argparse.ArgumentParser(description="从 COLMAP TXT 模型转换为 3DGS `cameras.json`。")
    parser.add_argument("--source_path", type=str, required=True, help="COLMAP 项目根目录。")
    args = parser.parse_args()
    convert_txt_to_3dgs(args.source_path)

if __name__ == "__main__":
    main()