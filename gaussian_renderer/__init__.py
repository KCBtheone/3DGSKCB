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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # 如果场景中没有高斯点，返回一个空背景以避免崩溃
    if pc.get_xyz.shape[0] == 0:
        h, w = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
        empty_render = bg_color.repeat(h, w, 1).permute(2, 0, 1)
        return {
            "render": empty_render,
            "viewspace_points": torch.empty(0, 3, device="cuda"),
            "visibility_filter": torch.empty(0, dtype=torch.bool, device="cuda"),
            "radii": torch.empty(0, device="cuda"),
            "depth": torch.zeros((1, h, w), device="cuda"),
        }

    # [核心对齐] 创建一个与3D高斯点一一对应的2D点张量，
    # 这是官方实现中将梯度从2D视图空间传回3D高斯位置的关键。
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 设置光栅化器
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing # 使用来自pipeline参数的设置
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 准备高斯属性
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    
    # 准备颜色/球谐系统
    if override_color is None:
        shs = pc.get_features
        colors_precomp = None
    else:
        # 如果提供了覆盖颜色(例如，用于渲染法线图)，则不使用球谐系数
        shs = None
        colors_precomp = override_color.float()

    # 核心光栅化
    rendered_image, radii, rendered_depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = None
    )
    
    # 整理输出
    rendered_image = rendered_image.clamp(0, 1)
    
    # 确定哪些高斯点在视锥体内且半径大于0（即对最终图像有贡献）
    visibility_filter = radii > 0
    
    # 返回一个字典，包含渲染结果和用于训练（特别是致密化）所需的信息
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points, # [核心对齐] 返回梯度占位符本身
        "visibility_filter": visibility_filter,
        "radii": radii,
        "depth": rendered_depth.unsqueeze(0),
    }