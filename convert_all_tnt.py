#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
convert_all_tnt.py (v3.0 - Self-Contained Cleanup & Conversion)

This script provides a one-stop solution for preparing the Tanks and Temples (T&T)
dataset for the 3D Gaussian Splatting pipeline.

Features:
- Hardcoded Path: The root directory for the T&T dataset is set internally.
- Automated Cleanup: Before conversion, it recursively finds and DELETES all
  'sparse' directories and 'database.db' files within the T&T scenes to ensure
  a fresh start and prevent errors from previous runs.
- Robust Conversion: It then processes all scenes, converting camera parameters
  into the correct COLMAP text format.
- Correct Naming: Ensures image names in 'images.txt' are pure filenames without
  any path prefixes, fixing the critical 'image does not exist in database' error.
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
from scipy.spatial.transform import Rotation
import shutil

# ==============================================================================
#                      [ 1. CONFIGURATION ]
# ==============================================================================
# --- [!! IMPORTANT !!] ---
# Please verify this path is correct for your system.
# This should be the absolute path to the 'tankandtemples' directory.
TNT_ROOT_DIRECTORY = "/root/autodl-tmp/gaussian-splatting/data/tankandtemples"
# ------------------------------------------------------------------------------


def parse_cam_file(filepath):
    """Parses a T&T _cam.txt file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    extrinsic = np.array([list(map(float, line.strip().split())) for line in lines[1:5]])
    intrinsic = np.array([list(map(float, line.strip().split())) for line in lines[7:10]])
    return intrinsic, extrinsic

def world_to_colmap(extrinsic_mat):
    """Converts a T&T world-to-camera extrinsic to a COLMAP camera-to-world pose."""
    R_w2c, t_w2c = extrinsic_mat[:3, :3], extrinsic_mat[:3, 3]
    R_c2w, t_c2w = R_w2c.T, -R_w2c.T @ t_w2c
    quat = Rotation.from_matrix(R_c2w).as_quat()
    return np.array([quat[3], quat[0], quat[1], quat[2]]), t_c2w

def cleanup_scene(scene_path: Path):
    """Deletes old sparse directory and database file for a given scene."""
    sparse_dir = scene_path / "sparse"
    db_file = scene_path / "database.db"
    
    if sparse_dir.exists() and sparse_dir.is_dir():
        print(f"  -> Deleting old directory: {sparse_dir}")
        shutil.rmtree(sparse_dir)
        
    if db_file.exists() and db_file.is_file():
        print(f"  -> Deleting old file: {db_file}")
        os.remove(db_file)

def process_scene(scene_path: Path):
    """Processes a single T&T scene and creates the COLMAP text files."""
    print(f"\n--- Processing Scene: {scene_path.name} ---")
    
    # 1. Cleanup old results first
    cleanup_scene(scene_path)

    # 2. Define paths and check for source data
    images_dir = scene_path / "images"
    cams_dir = scene_path / "cams"
    output_dir = scene_path / "sparse" / "0"

    if not images_dir.is_dir() or not cams_dir.is_dir():
        print(f"  -> ⚠️  Skipping: Source 'images' or 'cams' folder not found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Read and convert data
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    colmap_cameras = {}
    colmap_images = []
    
    for idx, image_name in enumerate(tqdm(image_files, desc=f"  -> Converting '{scene_path.name}'")):
        image_id = idx + 1
        try:
            img = cv2.imread(str(images_dir / image_name))
            height, width, _ = img.shape
            K, extrinsic_mat = parse_cam_file(cams_dir / f"{Path(image_name).stem}_cam.txt")
        except Exception as e:
            print(f"  -> ⚠️  Warning: Could not process {image_name}. Error: {e}. Skipping.")
            continue
            
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        cam_params_key = (width, height, fx, fy, cx, cy)
        
        if cam_params_key not in colmap_cameras:
            colmap_cameras[cam_params_key] = len(colmap_cameras) + 1
        camera_id = colmap_cameras[cam_params_key]
        
        qvec, tvec = world_to_colmap(extrinsic_mat)
        
        # Correctly formatted image entry with pure filename
        image_entry = (
            f"{image_id} {qvec[0]:.8f} {qvec[1]:.8f} {qvec[2]:.8f} {qvec[3]:.8f} "
            f"{tvec[0]:.8f} {tvec[1]:.8f} {tvec[2]:.8f} {camera_id} {image_name}\n"
        )
        colmap_images.append(image_entry)
        colmap_images.append("\n")

    # 4. Write new COLMAP text files
    with open(output_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(colmap_cameras)}\n")
        sorted_cameras = sorted(colmap_cameras.items(), key=lambda item: item[1])
        for params, cam_id in sorted_cameras:
            width, height, fx, fy, cx, cy = params
            f.write(f"{cam_id} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    with open(output_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_files)}\n")
        f.writelines(colmap_images)

    with open(output_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 0\n")
        
    print(f"  -> ✅  Successfully created fresh COLMAP files in '{output_dir}'.")

def main():
    """Main function to find, clean, and process all T&T scenes."""
    root_path = Path(TNT_ROOT_DIRECTORY)
    
    if not root_path.is_dir():
        print(f"❌ FATAL ERROR: The specified directory does not exist: '{root_path}'")
        print("Please correct the 'TNT_ROOT_DIRECTORY' variable at the top of this script.")
        sys.exit(1)
        
    print(f"🚀  Starting Cleanup and Conversion for all scenes in '{root_path}'...")
    
    # Find all valid scene directories (must contain 'images' and 'cams' subfolders)
    scenes_to_process = [d for d in root_path.rglob('*') if (d / 'images').is_dir() and (d / 'cams').is_dir()]
    
    if not scenes_to_process:
        print("❌ WARNING: No valid T&T scene directories were found.")
        print("A valid scene must contain both an 'images' and a 'cams' subfolder.")
        sys.exit(1)
        
    print(f"Found {len(scenes_to_process)} scenes to process.")
    
    for scene_path in scenes_to_process:
        process_scene(scene_path)
        
    print("\n🎉  All Tanks and Temples scenes have been cleaned and converted successfully!")
    print("You can now run the main benchmark script.")

if __name__ == "__main__":
    main()