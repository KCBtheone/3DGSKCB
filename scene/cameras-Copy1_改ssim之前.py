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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    # [核心修改] 在构造函数签名中加入 gt_confidence_map
    def __init__(self, colmap_id, R, T, FoVx, FoVy,
                 image, image_name, uid,
                 gt_alpha_mask=None,
                 gt_normal_map=None,
                 gt_depth_map=None,
                 gt_confidence_map=None, # <--- 新增参数
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.data_device = data_device

        # 保留你的诊断性打印
        # print(f"[DEBUG] CameraUID {uid}: Received image tensor with shape={image.shape}.")

        self.original_image = image.to(self.data_device)
        self.image_height = self.original_image.shape[1]
        self.image_width = self.original_image.shape[2]
        
        # print(f"[DEBUG] CameraUID {uid}: Set self.image_width={self.image_width}, self.image_height={self.image_height}.")

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)

        # [核心修改] 存储所有几何先验
        self.gt_normal_map = gt_normal_map.to(self.data_device) if gt_normal_map is not None else None
        self.gt_depth_map = gt_depth_map.to(self.data_device) if gt_depth_map is not None else None
        self.gt_confidence_map = gt_confidence_map.to(self.data_device) if gt_confidence_map is not None else None # <--- 新增存储逻辑
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    """一个用于高效存储和传输的轻量级相机类。"""
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]