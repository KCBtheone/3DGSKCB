# ==============================================================================
#           /scene/dataset_readers.py (高性能优化版)
# ==============================================================================
# 优化算法说明:
# 1. 原始 find_priors 函数的算法复杂度为 O(N*k)，其中 N 是相机数量，k 是
#    需要检查的先验文件类型数量 (通常是3)。每次检查都是一次文件系统I/O调用。
# 2. 优化后的 find_priors_optimized 函数采用了预处理和哈希查找的策略：
#    a. 首先，通过一次 O(M) 的目录扫描 (M为先验目录中的文件总数)，将所有
#       存在的文件名读入内存，并构建一个哈希映射 (字典)。
#    b. 其次，遍历 N 个相机，对每个相机，在哈希映射中进行 O(1) 的常数时间
#       查找来匹配对应的先验文件。
# 3. 最终算法复杂度近似为 O(M+N)，将多次I/O操作减少为一次，从而极大地
#    提升了在大量相机或慢速文件系统下的执行效率。
# ==============================================================================

import os
import sys
from pathlib import Path
from typing import NamedTuple
import numpy as np
import json
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.sh_utils import SH2RGB
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_points3D_text as read_points3d_from_colmap_loader
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
    gt_depth_map_path: str = None
    gt_confidence_map_path: str = None

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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(sorted(cam_extrinsics.keys())):
        extr = cam_extrinsics[key]
        if extr.camera_id not in cam_intrinsics:
            continue
        intr = cam_intrinsics[extr.camera_id]
        height, width = intr.height, intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY, FovX = focal2fov(focal_length_x, height), focal2fov(focal_length_x, width)
        elif intr.model in ["PINHOLE", "RADIAL", "OPENCV", "FULL_OPENCV"]:
            focal_length_x, focal_length_y = intr.params[0], intr.params[1]
            FovY, FovX = focal2fov(focal_length_y, height), focal2fov(focal_length_x, width)
        else:
            focal_length_x, focal_length_y = intr.params[0], intr.params[1]
            FovY, FovX = focal2fov(focal_length_y, height), focal2fov(focal_length_x, width)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(extr.name)
        if not os.path.exists(image_path):
            continue
        cam_infos.append(CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                                      image_path=image_path, image_name=image_name, width=width, height=height))
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    if 'nx' in vertices.data.dtype.names:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, (rgb * 255).astype(np.uint8)), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(path)

# ==============================================================================
#                       ‼️‼️ [核心优化区] ‼️‼️
# ==============================================================================
def find_priors_optimized(cam_info_list, base_path):
    """
    高效地查找几何先验文件。
    该版本只扫描一次目录，然后使用快速的集合查找来匹配文件。
    """
    priors_dir = Path(base_path) / "geometry_priors"
    if not priors_dir.exists():
        print(f"INFO: Geometry priors directory not found at '{priors_dir}'. Skipping prior loading.")
        return cam_info_list
    
    print(f"INFO: Scanning for geometry priors in '{priors_dir}'...")
    
    # 算法步骤 1: 一次性I/O操作，构建内存哈希映射
    # - priors_map: {'image_base_name': {'normal': '/path/to/normal.png', 'depth': ...}}
    priors_map = {}
    for f in priors_dir.iterdir():
        if f.is_file():
            stem = f.stem
            if stem.endswith('_normal'):
                base_name = stem[:-7]
                if base_name not in priors_map: priors_map[base_name] = {}
                priors_map[base_name]['normal'] = str(f)
            elif stem.endswith('_confidence'):
                base_name = stem[:-11]
                if base_name not in priors_map: priors_map[base_name] = {}
                priors_map[base_name]['confidence'] = str(f)
            else: # 假设是深度图
                base_name = stem
                if base_name not in priors_map: priors_map[base_name] = {}
                priors_map[base_name]['depth'] = str(f)

    print(f"INFO: Scan complete. Found priors for {len(priors_map)} unique images.")
    
    # 算法步骤 2: 循环内只进行 O(1) 的哈希查找
    new_list = []
    for cam in tqdm(cam_info_list, desc="Matching geometry priors"):
        img_basename = Path(cam.image_path).stem
        
        found_priors = priors_map.get(img_basename, {})
        
        new_list.append(cam._replace(
            gt_normal_map_path=found_priors.get('normal'),
            gt_depth_map_path=found_priors.get('depth'),
            gt_confidence_map_path=found_priors.get('confidence')
        ))
        
    return new_list

def readColmapScene(path, images, eval, llffhold=8):
    sparse_folder_path = Path(path) / "sparse"
    if (sparse_folder_path / "0").exists():
        sparse_path = sparse_folder_path / "0"
    else:
        sparse_path = sparse_folder_path

    print("INFO: Loading COLMAP model from .txt files...")
    cam_extrinsics = read_extrinsics_text(str(sparse_path / "images.txt"))
    cam_intrinsics = read_intrinsics_text(str(sparse_path / "cameras.txt"))
    
    reading_dir = "images" if images is None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted, key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for i, c in enumerate(cam_infos) if i % llffhold != 0]
        test_cam_infos = [c for i, c in enumerate(cam_infos) if i % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = str(sparse_path / "points3D.ply")
    if not os.path.exists(ply_path):
        print("INFO: points3D.ply not found. Creating it from points3D.txt...")
        try:
            xyz, rgb, _ = read_points3d_from_colmap_loader(str(sparse_path / "points3D.txt"))
            storePly(ply_path, xyz, rgb / 255.0)
        except Exception as e:
            storePly(ply_path, np.zeros((0,3)), np.zeros((0,3)))
    
    pcd = fetchPly(ply_path)

    # 算法步骤 3: 调用优化后的函数
    all_cam_infos = train_cam_infos + test_cam_infos
    all_cam_infos_with_priors = find_priors_optimized(all_cam_infos, path)
    
    # 重新分割训练集和测试集
    train_cam_infos = all_cam_infos_with_priors[:len(train_cam_infos)]
    test_cam_infos = all_cam_infos_with_priors[len(train_cam_infos):]

    return SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization, ply_path=ply_path)

# ... (Blender 加载部分保持不变)
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, desc=f"Loading {transformsfile}")):
            image_path = os.path.join(path, frame["file_path"] + extension)
            try:
                pil_image = Image.open(image_path)
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
    if not os.path.exists(ply_path):
        num_pts = 100_000
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)
    return SceneInfo(point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization, ply_path=ply_path)

sceneLoadTypeCallbacks = {
    "Colmap": readColmapScene,
    "Blender": readBlenderScene
}