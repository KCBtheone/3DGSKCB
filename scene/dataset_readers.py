"""
Copyright (C) 2023, Inria
GRAPHDECO research group, https://team.inria.fr/graphdeco
All rights reserved.
This software is free for non-commercial, research and evaluation use
under the terms of the LICENSE.md file.
For inquiries contact  george.drettakis@inria.fr
"""

import os
import sys
from pathlib import Path
from typing import NamedTuple, Dict, List
import numpy as np
import json
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB
from scene.colmap_loader import read_colmap_model, qvec2rotmat, Camera as ColmapCamera, Image as ColmapImage
from utils.graphics_utils import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    gt_normal_map_path: str = None
    gt_confidence_map_path: str = None
    gt_curvature_map_path: str = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = np.array([0.0, 0.0, 0.0])

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    try:
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.zeros_like(positions)
        if 'nx' in vertices.data.dtype.names:
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        return BasicPointCloud(points=positions, colors=colors, normals=normals)
    except Exception as e:
        return None

def storePly(path, xyz, rgb):
    if xyz.shape[0] == 0:
        el = PlyElement.describe(np.empty(0, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
        PlyData([el]).write(path)
        return
        
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, (rgb * 255).astype(np.uint8)), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(path)

def find_priors_optimized(cam_info_list, base_path):
    priors_dir = Path(base_path) / "priors"
    if not priors_dir.exists():
        return cam_info_list
    
    print(f"INFO: Scanning for geometry priors in '{priors_dir}'...")
    priors_map = {}
    for f in priors_dir.iterdir():
        if f.is_file():
            stem = f.stem
            if stem.endswith('_normal'): base_name, p_type = stem[:-7], 'normal'
            elif stem.endswith('_confidence'): base_name, p_type = stem[:-11], 'confidence'
            elif stem.endswith('_curvature'): base_name, p_type = stem[:-10], 'curvature'
            else: continue
            
            if base_name not in priors_map: priors_map[base_name] = {}
            priors_map[base_name][p_type] = str(f)
            
    new_list = []
    for cam in tqdm(cam_info_list, desc="Matching geometry priors"):
        img_basename = Path(cam.image_path).stem
        found_priors = priors_map.get(img_basename, {})
        new_list.append(cam._replace(
            gt_normal_map_path=found_priors.get('normal'),
            gt_confidence_map_path=found_priors.get('confidence'),
            gt_curvature_map_path=found_priors.get('curvature')
        ))
    return new_list

def readColmapCameras(images: Dict[int, ColmapImage], cameras: Dict[int, ColmapCamera], images_folder: str) -> List[CameraInfo]:
    cam_infos = []
    for idx, key in enumerate(sorted(images.keys())):
        extr = images[key]
        if extr.camera_id not in cameras:
            continue
        intr = cameras[extr.camera_id]
        height, width = intr.height, intr.width
        uid = extr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length = intr.params[0]
            FovY = focal2fov(focal_length, height)
            FovX = focal2fov(focal_length, width)
        elif intr.model == "PINHOLE":
            focal_length_x, focal_length_y = intr.params[0], intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            focal_length_x, focal_length_y = intr.params[0], intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(extr.name)
        if not os.path.exists(image_path):
            continue

        cam_infos.append(CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                                    image_path=image_path, image_name=image_name,
                                    width=width, height=height))
    return cam_infos

def readColmapScene(path, images_folder_name, eval, llffhold=8):
    sparse_folder = os.path.join(path, "sparse")
    if os.path.exists(os.path.join(sparse_folder, "0")):
        sparse_folder = os.path.join(sparse_folder, "0")
    
    bin_path = os.path.join(sparse_folder, "cameras.bin")
    txt_path = os.path.join(sparse_folder, "cameras.txt")

    if os.path.exists(bin_path):
        model_ext = ".bin"
        print(f"INFO: Loading COLMAP model from .bin files in '{sparse_folder}'...")
    elif os.path.exists(txt_path):
        model_ext = ".txt"
        print(f"INFO: Loading COLMAP model from .txt files in '{sparse_folder}'...")
    else:
        raise FileNotFoundError(f"Could not find COLMAP model files (cameras.bin or cameras.txt) in {sparse_folder}")

    cameras, images_data, points3D = read_colmap_model(sparse_folder, ext=model_ext)
    
    cam_infos_unsorted = readColmapCameras(images=images_data, cameras=cameras, images_folder=os.path.join(path, images_folder_name))
    cam_infos = sorted(cam_infos_unsorted, key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for i, c in enumerate(cam_infos) if i % llffhold != 0]
        test_cam_infos = [c for i, c in enumerate(cam_infos) if i % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(sparse_folder, "points3D.ply")
    if len(points3D) > 0:
        xyz = np.array([p.xyz for p in points3D.values()])
        rgb = np.array([p.rgb for p in points3D.values()])
    else:
        xyz = np.zeros((0, 3))
        rgb = np.zeros((0, 3))
    
    storePly(ply_path, xyz, rgb / 255.0)
    pcd = BasicPointCloud(points=xyz, colors=rgb / 255., normals=np.zeros((xyz.shape[0], 3)))
    
    all_cam_infos = train_cam_infos + test_cam_infos
    all_cam_infos_with_priors = find_priors_optimized(all_cam_infos, path)
    train_cam_infos = all_cam_infos_with_priors[:len(train_cam_infos)]
    test_cam_infos = all_cam_infos_with_priors[len(train_cam_infos):]

    return SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization, ply_path=ply_path)

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, desc=f"Loading {transformsfile}")):
            image_path = os.path.join(path, frame["file_path"] + extension)
            if not os.path.exists(image_path): continue
            
            try:
                with Image.open(image_path) as pil_image:
                    width, height = pil_image.size
            except IOError:
                continue

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1
            
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            fovy = focal2fov(fov2focal(fovx, width), height)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                                        image_path=image_path, image_name=Path(image_path).stem,
                                        width=width, height=height))
    return cam_infos

def readBlenderScene(path, white_background, eval):
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background)
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    ply_path = os.path.join(path, "points3d.ply")
    pcd = None
    if os.path.exists(ply_path):
        pcd = fetchPly(ply_path)
    
    if pcd is None:
        num_pts = 100_000
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)

    all_cam_infos = train_cam_infos + test_cam_infos
    all_cam_infos_with_priors = find_priors_optimized(all_cam_infos, path)
    train_cam_infos = all_cam_infos_with_priors[:len(train_cam_infos)]
    test_cam_infos = all_cam_infos_with_priors[len(train_cam_infos):]

    return SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization, ply_path=ply_path)

def readTNTScene(path, eval):
    """
    Reads a scene from the Tanks and Temples (TNT) dataset format.
    """
    print("INFO: Reading TNT scene...")

    def read_tnt_cameras(path, image_names):
        cam_infos = []
        for idx, image_name in enumerate(tqdm(image_names, desc="Loading camera data")):
            image_stem = Path(image_name).stem
            
            # Construct paths, checking common directory names
            image_path = os.path.join(path, "rgb", image_name)
            cam_path = os.path.join(path, "pose", f"{image_stem}.txt")

            if not os.path.exists(image_path):
                image_path_alt = os.path.join(path, "images", image_name)
                if os.path.exists(image_path_alt): image_path = image_path_alt
                else:
                    print(f"WARNING: Could not find image file {image_name} in 'rgb/' or 'images/'. Skipping.")
                    continue
            
            if not os.path.exists(cam_path):
                cam_path_alt = os.path.join(path, "cams", f"{image_stem}.txt")
                if os.path.exists(cam_path_alt): cam_path = cam_path_alt
                else:
                    print(f"WARNING: Could not find camera file for {image_name} in 'pose/' or 'cams/'. Skipping.")
                    continue

            # Read image dimensions
            try:
                with Image.open(image_path) as pil_image:
                    width, height = pil_image.size
            except IOError as e:
                print(f"WARNING: Could not read image {image_path}: {e}. Skipping.")
                continue

            # Read camera parameters from .txt file
            try:
                with open(cam_path, "r") as f: lines = f.readlines()
                c2w = np.array([list(map(float, line.strip().split())) for line in lines[:4]])
                intrinsics = np.array([list(map(float, line.strip().split())) for line in lines[5:8]])
                fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            except (IOError, ValueError) as e:
                print(f"WARNING: Could not read or parse camera file {cam_path}: {e}. Skipping.")
                continue

            # Convert from OpenCV (right, down, forward) to NeRF/OpenGL (right, up, backward)
            c2w[:3, 1:3] *= -1

            # Get world-to-camera matrix and extract R, T, following existing convention
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            # Convert intrinsics to Field of View
            FovY = focal2fov(fy, height)
            FovX = focal2fov(fx, width)
            
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                                      image_path=image_path, image_name=image_name,
                                      width=width, height=height))
        return cam_infos

    # Read train/test image lists
    try:
        with open(os.path.join(path, "train_list.txt"), "r") as f:
            train_image_names = [line.strip() for line in f.readlines()]
        with open(os.path.join(path, "test_list.txt"), "r") as f:
            test_image_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError as e:
        raise Exception(f"Could not find train_list.txt or test_list.txt in {path}. Error: {e}")

    train_cam_infos = read_tnt_cameras(path, train_image_names)
    test_cam_infos = read_tnt_cameras(path, test_image_names)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # Locate and load point cloud from COLMAP's sparse reconstruction
    ply_path, pcd = "", None
    sparse_folder = os.path.join(path, "sparse")
    if os.path.exists(sparse_folder):
        if os.path.exists(os.path.join(sparse_folder, "0")):
             sparse_folder = os.path.join(sparse_folder, "0")
        ply_path_candidate = os.path.join(sparse_folder, "points3D.ply")
        if os.path.exists(ply_path_candidate): ply_path = ply_path_candidate

    if not ply_path and os.path.exists(os.path.join(path, "points3D.ply")):
        ply_path = os.path.join(path, "points3D.ply")

    if ply_path:
        print(f"INFO: Loading point cloud from {ply_path}")
        pcd = fetchPly(ply_path)
    else:
        print("WARNING: Could not find points3D.ply. Generating a random point cloud.")
        num_pts = 100_000
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        ply_path = os.path.join(path, "points3D_random.ply")
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    
    if pcd is None: raise Exception("Could not load or create a point cloud.")
        
    all_cam_infos = train_cam_infos + test_cam_infos
    all_cam_infos_with_priors = find_priors_optimized(all_cam_infos, path)
    train_cam_infos = all_cam_infos_with_priors[:len(train_cam_infos)]
    test_cam_infos = all_cam_infos_with_priors[len(train_cam_infos):]

    return SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization, ply_path=ply_path)

sceneLoadTypeCallbacks = {
    "Colmap": readColmapScene,
    "Blender": readBlenderScene,
    "TNT": readTNTScene
}