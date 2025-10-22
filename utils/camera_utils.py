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

import torch
from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
import json
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

def load_priors(cam_info):
    """辅助函数，用于加载所有可用的几何先验图。"""
    
    # 辅助函数，用于加载单个图像文件
    def _load_img(path, is_single_channel=False):
        if path is None or not os.path.exists(path):
            return None
        try:
            img = Image.open(path)
            if is_single_channel and img.mode != 'L':
                img = img.convert('L')
            return TF.to_tensor(img)
        except Exception as e:
            print(f"WARNING: Could not load prior image at {path}. Reason: {e}")
            return None

    # 加载所有先验图
    gt_normal_map = _load_img(cam_info.gt_normal_map_path)
    gt_confidence_map = _load_img(cam_info.gt_confidence_map_path, is_single_channel=True)
    gt_curvature_map = _load_img(cam_info.gt_curvature_map_path, is_single_channel=True)
    
    return gt_normal_map, gt_confidence_map, gt_curvature_map

def loadCam(args, id, cam_info, resolution_scale):
    """从CameraInfo对象加载一个完整的Camera对象。"""
    
    # 加载原始RGB图像
    orig_w, orig_h = cam_info.width, cam_info.height
    
    if args.resolution in [1, 2, 4, 8]:
        resolution = (orig_w // args.resolution, orig_h // args.resolution)
    else:
        # 如果resolution是-1，则使用原始分辨率
        resolution = (orig_w, orig_h)

    try:
        # [鲁棒性修改] 使用 with open(...) 确保文件被正确关闭
        with Image.open(cam_info.image_path) as img:
            # 调整图像大小并转换为Tensor
            resized_image_rgb = TF.to_tensor(img.resize(resolution, Image.LANCZOS))
    except Exception as e:
        print(f"ERROR: Failed to load and resize image: {cam_info.image_path}. Reason: {e}")
        # 返回None，让上层调用者可以跳过这个损坏的相机
        return None

    # ‼️‼️ [核心修复] ‼️‼️
    # 1. 移除对 gt_depth_map_path 的引用。
    # 2. 调用新的辅助函数来加载所有我们关心的先验图。
    gt_normal_map, gt_confidence_map, gt_curvature_map = load_priors(cam_info)

    # 创建并返回一个完整的Camera对象
    # 参数顺序必须与 scene/cameras.py 中的 Camera.__init__ 完全匹配
    return Camera(colmap_id=cam_info.uid,
                  R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=resized_image_rgb,
                  image_name=cam_info.image_name,
                  uid=id,
                  gt_normal_map=gt_normal_map,
                  gt_confidence_map=gt_confidence_map,
                  gt_curvature_map=gt_curvature_map # <--- 传递曲率图
                 )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    """将CameraInfo列表转换为Camera对象列表。"""
    camera_list = []
    
    # 使用tqdm显示加载进度
    for id, c in enumerate(tqdm(cam_infos, desc="Loading cameras")):
        loaded_cam = loadCam(args, id, c, resolution_scale)
        if loaded_cam: # 只有当相机成功加载时才添加到列表
            camera_list.append(loaded_cam)
            
    return camera_list

def camera_to_JSON(id, camera : Camera):
    """将Camera对象转换为JSON格式，供viewer使用。"""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.T
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.image_width,
        'height' : camera.image_height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FoVy, camera.image_height),
        'fx' : fov2focal(camera.FoVx, camera.image_width)
    }
    return camera_entry