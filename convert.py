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

import os
import sys
from pathlib import Path
import json
import shutil
from PIL import Image
from typing import NamedTuple
import numpy as np
import argparse
from tqdm import tqdm

# --- Helper Functions ---

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))

# --- Main Conversion Logic ---

def convert_nerf_to_colmap(source_path, use_links=True):
    print(f"Converting NeRF data at '{source_path}' to COLMAP format...")
    
    # --- 1. Prepare Output Directories ---
    images_out_dir = os.path.join(source_path, "images")
    sparse_out_dir = os.path.join(source_path, "sparse", "0")

    print(f"  - Cleaning up old directories...")
    if os.path.exists(images_out_dir):
        shutil.rmtree(images_out_dir)
    if os.path.exists(os.path.dirname(sparse_out_dir)):
        shutil.rmtree(os.path.dirname(sparse_out_dir))

    os.makedirs(images_out_dir)
    os.makedirs(sparse_out_dir)

    # --- 2. Process JSON and Copy/Link Images ---
    cam_infos = []
    processed_images = set() # 用来跟踪已经处理过的图片名
    focal_length = 0
    width, height = 0, 0

    for split in ["train", "test", "val"]:
        json_path = os.path.join(source_path, f"transforms_{split}.json")
        if not os.path.exists(json_path):
            print(f"  - WARNING: '{json_path}' not found, skipping.")
            continue

        print(f"  - Processing '{split}' split...")
        with open(json_path, 'r') as f:
            meta = json.load(f)

        if not width and meta["frames"]:
            first_frame_path = os.path.join(source_path, meta["frames"][0]["file_path"] + '.png')
            if os.path.exists(first_frame_path):
                 with Image.open(first_frame_path) as img:
                    width, height = img.size
                 focal_length = fov2focal(meta["camera_angle_x"], width)

        for frame in tqdm(meta["frames"], desc=f"    > Reading {split} frames"):
            img_path_in = os.path.join(source_path, frame["file_path"] + '.png')
            img_name = os.path.basename(img_path_in)
            
            # ============================ [ ⭐⭐⭐ 核心修复 ⭐⭐⭐ ] ============================
            # 只有当图片名没有被处理过时，才创建链接和相机信息
            if img_name in processed_images:
                continue
            # =================================================================================

            img_path_out = os.path.join(images_out_dir, img_name)
            
            if os.path.exists(img_path_in):
                if use_links:
                    os.symlink(os.path.relpath(img_path_in, images_out_dir), img_path_out)
                else:
                    shutil.copy(img_path_in, img_path_out)
            else:
                print(f"    > WARNING: Source image not found: {img_path_in}")
                continue
            
            processed_images.add(img_name) # 标记为已处理
                
            c2w = np.array(frame["transform_matrix"])
            w2c = np.linalg.inv(c2w)
            w2c[0:3, 1:3] *= -1 
            R = w2c[:3,:3]
            T = w2c[:3,3]
            
            cam_infos.append({"name": img_name, "R": R, "T": T})

    if not cam_infos:
        print("FATAL: No camera information was processed. Exiting.")
        return

    # --- 3. Write COLMAP files ---
    print("  - Writing COLMAP sparse files...")
    with open(os.path.join(sparse_out_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {width} {height} {focal_length} {focal_length} {width/2} {height/2}\n")
    
    with open(os.path.join(sparse_out_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[]\n")
        f.write(f"# Number of images: {len(cam_infos)}\n")
        
        for idx, cam in enumerate(cam_infos):
            qvec = rotmat2qvec(cam["R"])
            tvec = cam["T"]
            f.write(f"{idx+1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} 1 {cam['name']}\n\n")

    with open(os.path.join(sparse_out_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        
    print("✅ Conversion complete!")
    print(f"  - Images are in: '{images_out_dir}'")
    print(f"  - Sparse model is in: '{sparse_out_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NeRF synthetic to COLMAP converter for benchmark scripts")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="Path to the NeRF synthetic dataset (e.g., data/nerf_synthetic/chair)")
    args = parser.parse_args()

    original_sparse_dir = os.path.join(args.source_path, "sparse")
    if os.path.exists(original_sparse_dir) and not os.path.islink(original_sparse_dir):
        backup_sparse_dir = os.path.join(args.source_path, "sparse_original")
        print(f"Backing up existing 'sparse' directory to '{backup_sparse_dir}'...")
        if os.path.exists(backup_sparse_dir):
            shutil.rmtree(backup_sparse_dir)
        os.rename(original_sparse_dir, backup_sparse_dir)
        
    convert_nerf_to_colmap(args.source_path)