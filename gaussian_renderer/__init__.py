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


    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass


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
        antialiasing=pipe.antialiasing 
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    

    if override_color is None:
        shs = pc.get_features
        colors_precomp = None
    else:
       
        shs = None
        colors_precomp = override_color.float()


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
    

    rendered_image = rendered_image.clamp(0, 1)
    

    visibility_filter = radii > 0
    

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "depth": rendered_depth.unsqueeze(0),
    }