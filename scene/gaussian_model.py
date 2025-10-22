#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved#
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
            self._scaling, self._rotation, self._opacity,
            self.max_radii2D, self.xyz_gradient_accum, self.denom,
            self.optimizer.state_dict(), self.spatial_lr_scale)

    def restore(self, model_args, training_args):
        (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest,
         self._scaling, self._rotation, self._opacity,
         self.max_radii2D, xyz_gradient_accum, denom, opt_dict, self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self): return self.scaling_activation(self._scaling)
    @property
    def get_rotation(self): return self.rotation_activation(self._rotation)
    @property
    def get_xyz(self): return self._xyz
    @property
    def get_features(self): return torch.cat((self._features_dc, self._features_rest), dim=1)
    @property
    def get_opacity(self): return self.opacity_activation(self._opacity)

    @property
    def get_isotropy(self):
        scales = self.get_scaling
        s_max, _ = torch.max(scales, dim=1)
        s_min, _ = torch.min(scales, dim=1)
        return (s_min / (s_max + 1e-8)).unsqueeze(-1)

    @property
    def get_normals(self):
        scales = self.get_scaling
        smallest_scale_indices = torch.argmin(scales, dim=1)
        normals_local = F.one_hot(smallest_scale_indices, num_classes=3).float()
        rotations = build_rotation(self.get_rotation)
        normals_world = torch.bmm(rotations, normals_local.unsqueeze(-1)).squeeze(-1)
        return F.normalize(normals_world, p=2, dim=1)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # =================================================================================
    # >>> [ 🚀 最终修正区: 采用更保守的批次大小 ] <<<
    # =================================================================================

    def compute_smoothness_loss(self, k_nearest: int = 5, batch_size: int = 2048):
        """[策略 #2] [最终修正] 计算法线平滑度损失，使用更小的批次以避免OOM。"""
        xyz = self.get_xyz
        normals = self.get_normals
        num_points = xyz.shape[0]

        if num_points < k_nearest + 1:
            return torch.tensor(0.0, device=xyz.device)

        knn_indices_list = []
        weights_list = []

        with torch.no_grad():
            for i in range(0, num_points, batch_size):
                end = i + batch_size
                batch_xyz = xyz[i:end]
                dist_sq_batch = torch.cdist(batch_xyz, xyz, p=2)**2
                dist_k_sq, topk_indices = torch.topk(dist_sq_batch, k=k_nearest + 1, dim=1, largest=False, sorted=True)
                knn_indices_list.append(topk_indices[:, 1:])
                neighbor_dist = torch.sqrt(dist_k_sq[:, 1:])
                weights = (1.0 / (neighbor_dist + 1e-8))
                weights = F.normalize(weights, p=1, dim=1)
                weights_list.append(weights)

        knn_indices = torch.cat(knn_indices_list, dim=0)
        weights = torch.cat(weights_list, dim=0)
        
        source_normals = normals.unsqueeze(1).expand(-1, k_nearest, -1)
        neighbor_normals = normals[knn_indices]
        
        diff_normals_sq = torch.sum((source_normals - neighbor_normals)**2, dim=-1)
        smoothness_loss = torch.mean(torch.sum(weights * diff_normals_sq, dim=1))
        return smoothness_loss

    def geometric_inconsistency_check(self, k_nearest: int, inconsistency_threshold: float, batch_size: int = 2048):
        """[策略 #3] [最终修正] 检查几何不一致性，使用更小的批次。"""
        xyz = self.get_xyz
        normals = self.get_normals
        num_points = xyz.shape[0]

        if num_points <= k_nearest:
            return torch.zeros(num_points, dtype=torch.bool, device=xyz.device)
        
        avg_cos_sim_list = []
        with torch.no_grad():
            for i in range(0, num_points, batch_size):
                end = i + batch_size
                batch_xyz = xyz[i:end]
                batch_normals = normals[i:end]
                dist_sq_batch = torch.cdist(batch_xyz, xyz, p=2)**2
                _, topk_indices = torch.topk(dist_sq_batch, k=k_nearest + 1, dim=1, largest=False)
                knn_indices = topk_indices[:, 1:]
                source_normals = batch_normals.unsqueeze(1).expand(-1, k_nearest, -1)
                neighbor_normals = normals[knn_indices]
                cos_sim = F.cosine_similarity(source_normals, neighbor_normals, dim=-1)
                avg_cos_sim_list.append(torch.mean(cos_sim, dim=1))

        avg_cos_sim = torch.cat(avg_cos_sim_list, dim=0)
        return avg_cos_sim < inconsistency_threshold

    # (保留原有方法)
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree: self.active_sh_degree += 1
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]): l.append(f'f_dc_{i}')
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]): l.append(f'f_rest_{i}')
        l.append('opacity')
        for i in range(self._scaling.shape[1]): l.append(f'scale_{i}')
        for i in range(self._rotation.shape[1]): l.append(f'rot_{i}')
        return l
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((plydata.elements[0]["x"], plydata.elements[0]["y"], plydata.elements[0]["z"]), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = plydata.elements[0]["f_dc_0"]
        features_dc[:, 1, 0] = plydata.elements[0]["f_dc_1"]
        features_dc[:, 2, 0] = plydata.elements[0]["f_dc_2"]
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = plydata.elements[0][attr_name]
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = plydata.elements[0][attr_name]
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = plydata.elements[0][attr_name]
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[name] = group["params"][0]
        return optimizable_tensors
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
        return optimizable_tensors
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
        return optimizable_tensors
    def densification_postfix(self, tensors_dict):
        optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    def add_densification_stats(self, screenspace_points_grad, visibility_filter):
        self.xyz_gradient_accum[visibility_filter] += torch.norm(screenspace_points_grad[visibility_filter,:2], dim=-1, keepdim=True)
        self.denom[visibility_filter] += 1

    def densify_and_prune(self, opt, iteration, scene_extent, max_screen_size):
        grads = self.xyz_gradient_accum / (self.denom + 1e-7)
        grads[grads.isnan()] = 0.0

        use_geo_densify = (opt.use_geometric_densify and iteration >= opt.geo_densify_start_iter)
        geo_inconsistency_mask = None
        if use_geo_densify:
            geo_inconsistency_mask = self.geometric_inconsistency_check(
                k_nearest=10, 
                inconsistency_threshold=opt.geo_inconsistency_threshold
            )
            large_gaussian_threshold = opt.geo_densify_relative_size_threshold * scene_extent
            large_gaussians_mask = self.get_scaling.max(dim=1).values > large_gaussian_threshold
            grads[large_gaussians_mask & geo_inconsistency_mask] = opt.densify_grad_threshold * 1.5

        self.densify_and_clone(grads, opt.densify_grad_threshold, scene_extent)
        self.densify_and_split(grads, opt.densify_grad_threshold, scene_extent)

        prune_mask_opacity = (self.get_opacity < opt.min_opacity).squeeze()
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask_opacity = torch.logical_or(prune_mask_opacity, big_points_vs)

        if use_geo_densify:
            prune_mask_geo = geo_inconsistency_mask & (self.get_opacity < 0.1).squeeze()
            prune_mask_opacity = torch.logical_or(prune_mask_opacity, prune_mask_geo)

        self.prune_points(prune_mask_opacity)
        torch.cuda.empty_cache()

    def densify_and_clone(self, grads, grad_threshold, extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        d = {"xyz": new_xyz, "f_dc": new_features_dc, "f_rest": new_features_rest, "opacity": new_opacities,
             "scaling": new_scaling, "rotation": new_rotation}
        self.densification_postfix(d)
    def densify_and_split(self, grads, grad_threshold, extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        d = {"xyz": new_xyz, "f_dc": new_features_dc, "f_rest": new_features_rest, "opacity": new_opacity,
             "scaling": new_scaling, "rotation": new_rotation}
        self.densification_postfix(d)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)