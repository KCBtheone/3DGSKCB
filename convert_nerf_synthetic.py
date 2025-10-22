#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import argparse
import os
import sys
from pathlib import Path
import json
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", help="path to the nerf synthetic dataset")
    args = parser.parse_args()
    
    source_path = args.source_path
    
    # 1. 创建输出目录
    sparse_path = os.path.join(source_path, "sparse")
    os.makedirs(sparse_path, exist_ok=True)
    
    input_path = os.path.join(source_path, "input")
    os.makedirs(input_path, exist_ok=True)

    print("Copying images to 'input' directory for reference...")
    for split in ["train", "test", "val"]:
        json_file = os.path.join(source_path, f"transforms_{split}.json")
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for frame in data["frames"]:
            img_path = os.path.join(source_path, frame["file_path"] + ".png")
            if not os.path.exists(img_path):
                img_path = os.path.join(source_path, frame["file_path"])
            if os.path.exists(img_path):
                do_system(f"cp {img_path} {input_path}")
            else:
                 print(f"Warning: image {img_path} not found.")

    # 2. 读取所有 JSON 文件并整合相机信息
    all_frames = []
    camera_angle_x = None
    for split in ["train", "test", "val"]:
        json_file = os.path.join(source_path, f"transforms_{split}.json")
        with open(json_file, 'r') as f:
            data = json.load(f)
            if camera_angle_x is None:
                camera_angle_x = data["camera_angle_x"]
            all_frames.extend(data["frames"])
    
    # 3. 创建 cameras.txt
    cam_file = os.path.join(sparse_path, "cameras.txt")
    img_path = os.path.join(source_path, all_frames[0]["file_path"] + ".png")
    if not os.path.exists(img_path):
        img_path = os.path.join(source_path, all_frames[0]["file_path"])
        
    img = Image.open(img_path)
    W, H = img.size
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    with open(cam_file, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 SIMPLE_PINHOLE {W} {H} {focal} {W/2} {H/2}\n")

    # 4. 创建 images.txt
    img_file = os.path.join(sparse_path, "images.txt")
    with open(img_file, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[]\n")
        
        for idx, frame in enumerate(all_frames):
            c2w = np.array(frame["transform_matrix"])
            c2w[0:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R_mat = w2c[0:3, 0:3]
            T = w2c[0:3, 3]
            q = Rotation.from_matrix(R_mat).as_quat()
            q = np.array([q[3], q[0], q[1], q[2]])
            image_id = idx + 1
            camera_id = 1
            name = os.path.basename(frame["file_path"]) + ".png"
            if not os.path.exists(os.path.join(input_path, name)):
                name = os.path.basename(frame["file_path"])
            f.write(f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {T[0]} {T[1]} {T[2]} {camera_id} {name}\n\n")

    # 5. 创建空的 points3D.txt
    pts_file = os.path.join(sparse_path, "points3D.txt")
    with open(pts_file, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

    print(f"✅ NeRF to COLMAP text files conversion complete.")
    print(f"Data saved in: {sparse_path}")

    # =================================================================================
    # >>> [ 核心修改: 绕过 model_converter ] <<<
    # 直接将生成的 .txt 文件移动到 `sparse/0` 目录
    # =================================================================================
    print("\nSkipping binary conversion due to COLMAP version issues.")
    print("Moving .txt files directly to sparse/0...")

    if os.path.exists(os.path.join(source_path, "sparse/0")):
        do_system(f"rm -rf {os.path.join(source_path, 'sparse/0')}")

    os.makedirs(os.path.join(source_path, "sparse/0"), exist_ok=True)
    
    do_system(f"mv {os.path.join(sparse_path, 'cameras.txt')} {os.path.join(source_path, 'sparse/0/')}")
    do_system(f"mv {os.path.join(sparse_path, 'images.txt')} {os.path.join(source_path, 'sparse/0/')}")
    do_system(f"mv {os.path.join(sparse_path, 'points3D.txt')} {os.path.join(source_path, 'sparse/0/')}")
    
    # 清理临时目录
    do_system(f"rm -rf {sparse_path}")
    
    print("\n✅ Conversion complete. `sparse/0` now contains .txt files.")
    print("3DGS will use these text files as a fallback.")
    print("You can now run the training script.")