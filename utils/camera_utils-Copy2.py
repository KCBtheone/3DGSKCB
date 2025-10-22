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
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import numpy as np
from PIL import Image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    """
    此函数现在负责从路径加载图像以及所有几何先验（法线、深度、置信度）。
    """
    try:
        pil_image = Image.open(cam_info.image_path)
    except IOError:
        print(f"ERROR: Could not load image at {cam_info.image_path}. Skipping this camera.")
        return None

    orig_w, orig_h = pil_image.size

    # 保留你的诊断性打印
    # print(f"[DEBUG] CamID {id}: Received resolution_scale={resolution_scale}. Original size=({orig_w}, {orig_h}). Path: {cam_info.image_name}")

    res_scale = resolution_scale if resolution_scale > 0 else 1.0
    resized_w = max(1, int(orig_w / res_scale))
    resized_h = max(1, int(orig_h / res_scale))
    resolution = (resized_w, resized_h)
    
    # print(f"[DEBUG] CamID {id}: Calculated new size for torch tensor=({resized_w}, {resized_h}).")

    resized_image_rgb = PILtoTorch(pil_image, resolution)

    gt_alpha_mask = None
    if resized_image_rgb.shape[0] == 4:
        gt_alpha_mask = resized_image_rgb[3].unsqueeze(0)
        resized_image_rgb = resized_image_rgb[:3]

    # --- [核心修改] 升级几何先验加载逻辑 ---

    # 1. 加载法线图
    gt_normal_map = None
    if cam_info.gt_normal_map_path:
        try:
            normal_pil = Image.open(cam_info.gt_normal_map_path)
            gt_normal_map = PILtoTorch(normal_pil, resolution)[:3]
        except IOError:
            print(f"WARNING: Could not load normal map at {cam_info.gt_normal_map_path}")

    # 2. 加载深度图
    gt_depth_map = None
    if cam_info.gt_depth_map_path:
        try:
            depth_pil = Image.open(cam_info.gt_depth_map_path)
            # 深度图可能是单通道或多通道，直接转换
            gt_depth_map = PILtoTorch(depth_pil, resolution)
        except IOError:
            print(f"WARNING: Could not load depth map at {cam_info.gt_depth_map_path}")

    # 3. 新增：加载置信度图
    gt_confidence_map = None
    if cam_info.gt_confidence_map_path:
        try:
            conf_pil = Image.open(cam_info.gt_confidence_map_path)
            # 置信度图是单通道灰度图，加载后归一化到 [0, 1]
            gt_confidence_map = PILtoTorch(conf_pil, resolution)
            if gt_confidence_map.max() > 1.0:
                gt_confidence_map = gt_confidence_map / 255.0
        except IOError:
            print(f"WARNING: Could not load confidence map at {cam_info.gt_confidence_map_path}")

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=resized_image_rgb,
                  image_name=cam_info.image_name,
                  uid=id,
                  gt_alpha_mask=gt_alpha_mask,
                  gt_normal_map=gt_normal_map,
                  gt_depth_map=gt_depth_map,
                  gt_confidence_map=gt_confidence_map, # 新增：传递置信度图
                  data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []
    for id, c in enumerate(cam_infos):
        loaded_cam = loadCam(args, id, c, resolution_scale)
        if loaded_cam is not None:
            camera_list.append(loaded_cam)
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
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