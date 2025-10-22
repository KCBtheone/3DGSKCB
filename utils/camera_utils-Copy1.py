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
import torch
import numpy as np
from utils.graphics_utils import fov2focal, getWorld2View2
from scene.cameras import Camera
from utils.general_utils import PILtoTorch
from PIL import Image

# ====================================================================================
#          >>> [ 核心修改 1: 修正 loadCam 函数签名 ] <<<
# ====================================================================================
# 新增 lines=None 和 opt=None 参数
def loadCam(args, id, cam_info, resolution_scale, lines=None, opt=None):
    
    # 匹配与当前相机视图对应的线段数据
    camera_lines = None
    if lines:
        # 使用 os.path.basename 来确保无论 cam_info.image_name 是相对路径还是绝对路径都能正确匹配
        image_base_name = os.path.basename(cam_info.image_name)
        camera_lines = lines.get(image_base_name)

    # 从文件加载全分辨率图像
    full_res_image = Image.open(cam_info.image_path)
    
    # 将PIL图像转换为PyTorch张量
    # 注意：PILtoTorch 应该返回一个未缩放的张量
    image_tensor = PILtoTorch(full_res_image, (cam_info.width, cam_info.height))

    # 分离颜色和 alpha 通道
    gt_image = image_tensor[:3, ...]
    gt_alpha_mask = None
    if image_tensor.shape[0] == 4:
        gt_alpha_mask = image_tensor[3:4, ...]
    
    # 根据 resolution_scale 对图像和alpha mask进行降采样
    if resolution_scale != 1.0:
        width = int(cam_info.width / resolution_scale)
        height = int(cam_info.height / resolution_scale)
        
        resized_gt_image = torch.nn.functional.interpolate(
            gt_image.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        if gt_alpha_mask is not None:
            resized_gt_alpha_mask = torch.nn.functional.interpolate(
                gt_alpha_mask.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
        else:
            resized_gt_alpha_mask = None
    else:
        resized_gt_image = gt_image
        resized_gt_alpha_mask = gt_alpha_mask

    # 将所有必需的参数，包括 resolution_scale, lines 和 opt, 传递给 Camera 构造函数
    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=resized_gt_image,
        gt_alpha_mask=resized_gt_alpha_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=args.data_device,
        lines=camera_lines,  # <-- 传递匹配到的线段数据
        opt=opt              # <-- 传递完整的优化参数对象
    )

# ====================================================================================
#          >>> [ 核心修改 2: 修正 cameraList_from_camInfos 函数签名 ] <<<
# ====================================================================================
# 新增 lines=None 和 opt=None 参数
def cameraList_from_camInfos(cam_infos, resolution_scale, args, lines=None, opt=None):
    camera_list = []

    for id, c in enumerate(cam_infos):
        # [核心修改 3] 将 lines 和 opt 传递给 loadCam
        camera_list.append(loadCam(args, id, c, resolution_scale, lines=lines, opt=opt))

    return camera_list

def camera_to_JSON(id, cam_info):
    w2c = getWorld2View2(cam_info.R, cam_info.T)
    c2w = np.linalg.inv(w2c)
    camera_center = c2w[:3, 3]

    cam_data = {
        'id' : id,
        'img_name' : cam_info.image_name,
        'width' : cam_info.width,
        'height' : cam_info.height,
        'position': camera_center.tolist(),
        'rotation': cam_info.R.tolist(),
        'fy' : fov2focal(cam_info.FovY, cam_info.height),
        'fx' : fov2focal(cam_info.FovX, cam_info.width)
    }
    return cam_data