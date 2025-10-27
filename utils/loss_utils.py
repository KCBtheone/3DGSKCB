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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from torchvision.models import vgg16
from torchvision.transforms.functional import rgb_to_grayscale

# Conditional import for wavelet loss
try:
    from pytorch_wavelets import DWTForward, DWTInverse
    WAVELET_IMPORTED = True
except ImportError:
    WAVELET_IMPORTED = False

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, reduction='mean'):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if reduction == 'mean':
        return ssim_map.mean()
    elif reduction == 'none':
        return ssim_map
    else:
        return ssim_map.mean()

def ssim(img1, img2, window_size=11, reduction='mean'):
    if img1.dim() == 3: img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device).type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, reduction=reduction)

def ssim_weighted(img1, img2, weights, window_size=11, reduction='mean'):
    if img1.dim() == 3: img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)
    if weights.dim() == 2: weights = weights.unsqueeze(0).unsqueeze(0)
    if weights.dim() == 3: weights = weights.unsqueeze(0)
    if weights.shape[1] == 1 and img1.shape[1] == 3:
        weights = weights.expand(-1, 3, -1, -1)
        
    ssim_map = _ssim(img1, img2, create_window(window_size, img1.shape[1]).to(img1.device), window_size, img1.shape[1], reduction='none')
    avg_pool = nn.AvgPool2d(window_size, stride=1, padding=window_size // 2)
    window_weights = avg_pool(weights)
    
    if window_weights.shape[-2:] != ssim_map.shape[-2:]:
        window_weights = F.interpolate(window_weights, size=ssim_map.shape[-2:], mode='bilinear', align_corners=False)

    if reduction == 'mean':
        return (ssim_map * window_weights).sum() / (window_weights.sum() + 1e-8)
    elif reduction == 'none':
        return ssim_map * window_weights

# ==============================================================================
#                 VGG16 感知损失 (Perceptual Loss) 模块
# ==============================================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True, device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        features = vgg16(pretrained=True).features.to(device).eval()
        self.slice1 = nn.Sequential(*features[0:4])
        self.slice2 = nn.Sequential(*features[4:9])
        self.slice3 = nn.Sequential(*features[9:16])
        self.slice4 = nn.Sequential(*features[16:23])
        for param in self.parameters():
            param.requires_grad = False
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        self.feature_layers = {'relu1_2': self.slice1, 'relu2_2': self.slice2, 'relu3_3': self.slice3, 'relu4_3': self.slice4}

    def get_features(self, x):
        x = (x - self.mean) / self.std
        if self.resize:
            x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        feats = {}
        h = self.slice1(x); feats['relu1_2'] = h
        h = self.slice2(h); feats['relu2_2'] = h
        h = self.slice3(h); feats['relu3_3'] = h
        h = self.slice4(h); feats['relu4_3'] = h
        return feats

    def forward(self, input_tensor, target_tensor, return_feats=False):
        if input_tensor.dim() == 3: input_tensor = input_tensor.unsqueeze(0)
        if target_tensor.dim() == 3: target_tensor = target_tensor.unsqueeze(0)
        pred_feats = self.get_features(input_tensor)
        with torch.no_grad():
            target_feats = self.get_features(target_tensor)
        if return_feats:
            return pred_feats, target_feats
        loss = sum(F.l1_loss(pred_feats[key], target_feats[key]) for key in pred_feats)
        return loss

# ==============================================================================
#       结构损失计算器 - 已集成所有高级算子
# ==============================================================================
class StructuralLoss(nn.Module):
    def __init__(self, opt):
        super(StructuralLoss, self).__init__()
        self.opt = opt
        self.device = 'cuda'

        mode = self.opt.structural_loss_mode

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device).reshape(1,1,3,3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device).reshape(1,1,3,3)
        self.sobel_x = sobel_x.repeat(3, 1, 1, 1)
        self.sobel_y = sobel_y.repeat(3, 1, 1, 1)

        scharr_x = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32, device=self.device).reshape(1,1,3,3)
        scharr_y = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32, device=self.device).reshape(1,1,3,3)
        self.scharr_x = scharr_x.repeat(3, 1, 1, 1)
        self.scharr_y = scharr_y.repeat(3, 1, 1, 1)


        if mode == 'log':
            self.log_kernel = self._create_log_kernel(self.opt.log_kernel_size, self.opt.log_sigma).to(self.device)
            self.log_kernel = self.log_kernel.repeat(3, 1, 1, 1)


        if mode == 'structure_tensor':
            pool_size = self.opt.struct_tensor_neighborhood_size
            self.neighborhood_pool = nn.AvgPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)


        if mode == 'wavelet':
            if not WAVELET_IMPORTED:
                raise ImportError("pytorch_wavelets is not installed. Please run 'pip install pytorch_wavelets' to use the wavelet loss.")
            self.dwt = DWTForward(J=self.opt.wavelet_levels, mode='symmetric', wave=self.opt.wavelet_type).to(self.device)
            self.iwt = DWTInverse(mode='symmetric', wave=self.opt.wavelet_type).to(self.device)

        if mode == 'pfg':
            self.perceptual_model = VGGPerceptualLoss(device=self.device)
    
    def forward(self, pred_im, gt_im):
        loss = torch.tensor(0.0, device=self.device)
        feedback_map = None
        mode = self.opt.structural_loss_mode
        if mode == 'none': 
            return loss, feedback_map

        pred_b = pred_im.unsqueeze(0)
        gt_b = gt_im.unsqueeze(0)

        if mode in ['sobel', 'ms_sobel', 'scharr', 'ms_scharr']:
            loss, feedback_map = self._gradient_based_loss(pred_b, gt_b, mode)
        elif mode == 'log':
            loss, feedback_map = self._log_loss(pred_b, gt_b)
        elif mode == 'structure_tensor':
            loss, feedback_map = self._structure_tensor_loss(pred_b, gt_b)
        elif mode == 'pfg':
            loss, feedback_map = self._pfg_loss(pred_im, gt_im) # pfg needs 3D tensor
        elif mode == 'struct_ssim':
            loss, feedback_map = self._struct_ssim_loss(pred_b, gt_b)
        elif mode == 'wavelet':
            loss, feedback_map = self._wavelet_loss(pred_b, gt_b)
            
        return loss, feedback_map

    def _gradient_based_loss(self, pred_b, gt_b, mode):
        is_multiscale = 'ms' in mode
        grad_op = 'scharr' if 'scharr' in mode else 'sobel'
        
        grad_x_kernel = self.scharr_x if grad_op == 'scharr' else self.sobel_x
        grad_y_kernel = self.scharr_y if grad_op == 'scharr' else self.sobel_y

        loss_total = 0.0
        num_scales = self.opt.ms_grad_scales if is_multiscale else 1
        scale_weights = [1.0 / (2**i) for i in range(num_scales)]

        for i in range(num_scales):
            current_pred_b = F.avg_pool2d(pred_b, 2**i) if i > 0 else pred_b
            current_gt_b = F.avg_pool2d(gt_b, 2**i) if i > 0 else gt_b
            
            grad_pred_x = F.conv2d(current_pred_b, grad_x_kernel, padding=1, groups=3)
            grad_pred_y = F.conv2d(current_pred_b, grad_y_kernel, padding=1, groups=3)
            with torch.no_grad():
                grad_gt_x = F.conv2d(current_gt_b, grad_x_kernel, padding=1, groups=3)
                grad_gt_y = F.conv2d(current_gt_b, grad_y_kernel, padding=1, groups=3)
            
            loss_total += scale_weights[i] * (l1_loss(grad_pred_x, grad_gt_x) + l1_loss(grad_pred_y, grad_gt_y))
        
        # Feedback map is always computed at the highest resolution
        grad_pred_x_h = F.conv2d(pred_b, grad_x_kernel, padding=1, groups=3)
        grad_pred_y_h = F.conv2d(pred_b, grad_y_kernel, padding=1, groups=3)
        with torch.no_grad():
            grad_gt_x_h = F.conv2d(gt_b, grad_x_kernel, padding=1, groups=3)
            grad_gt_y_h = F.conv2d(gt_b, grad_y_kernel, padding=1, groups=3)
        
        feedback_map = (torch.abs(grad_pred_x_h - grad_gt_x_h) + torch.abs(grad_pred_y_h - grad_gt_y_h)).sum(dim=1, keepdim=True)
        return loss_total, feedback_map

    def _create_log_kernel(self, kernel_size, sigma):
        # Create a 2D Gaussian kernel
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        
        mean = (kernel_size - 1) / 2.
        variance = sigma**2.
        
        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*np.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # Calculate the Laplacian of the Gaussian
        # The operator is: (x^2 + y^2 - 2*sigma^2) / sigma^4
        laplacian_term = ((torch.sum((xy_grid - mean)**2., dim=-1) - 2*variance) / variance**2)
        log_kernel = gaussian_kernel * laplacian_term
        
        # Normalize to make sure the kernel sums to zero
        log_kernel = log_kernel - log_kernel.mean()
        
        return log_kernel.unsqueeze(0).unsqueeze(0)

    def _log_loss(self, pred_b, gt_b):
        pred_response = F.conv2d(pred_b, self.log_kernel, padding=self.opt.log_kernel_size//2, groups=3)
        with torch.no_grad():
            gt_response = F.conv2d(gt_b, self.log_kernel, padding=self.opt.log_kernel_size//2, groups=3)
        
        error_map = torch.abs(pred_response - gt_response)
        loss = error_map.mean()
        feedback_map = error_map.sum(dim=1, keepdim=True)
        return loss, feedback_map
        
    def _structure_tensor_loss(self, pred_b, gt_b):

        Gx_pred, Gy_pred = F.conv2d(pred_b, self.sobel_x, padding=1, groups=3), F.conv2d(pred_b, self.sobel_y, padding=1, groups=3)
        with torch.no_grad():
            Gx_gt, Gy_gt = F.conv2d(gt_b, self.sobel_x, padding=1, groups=3), F.conv2d(gt_b, self.sobel_y, padding=1, groups=3)


        J_pred_xx, J_pred_yy, J_pred_xy = Gx_pred**2, Gy_pred**2, Gx_pred*Gy_pred
        J_gt_xx, J_gt_yy, J_gt_xy = Gx_gt**2, Gy_gt**2, Gx_gt*Gy_gt


        J_pred_xx, J_pred_yy, J_pred_xy = self.neighborhood_pool(J_pred_xx), self.neighborhood_pool(J_pred_yy), self.neighborhood_pool(J_pred_xy)
        J_gt_xx, J_gt_yy, J_gt_xy = self.neighborhood_pool(J_gt_xx), self.neighborhood_pool(J_gt_yy), self.neighborhood_pool(J_gt_xy)


        loss = l1_loss(J_pred_xx, J_gt_xx) + l1_loss(J_pred_yy, J_gt_yy) + l1_loss(J_pred_xy, J_gt_xy)

        feedback_map = (torch.abs(J_pred_xx - J_gt_xx) + torch.abs(J_pred_yy - J_gt_yy) + torch.abs(J_pred_xy - J_gt_xy)).sum(dim=1, keepdim=True)
        return loss, feedback_map

    def _pfg_loss(self, pred, gt):
        pred_feats, gt_feats = self.perceptual_model(pred, gt, return_feats=True)
        feat_map_pred = pred_feats[self.opt.pfg_feature_layer]
        feat_map_gt = gt_feats[self.opt.pfg_feature_layer]
        
        channels = feat_map_pred.shape[1]
        sobel_x_feat = self.sobel_x[0].repeat(channels, 1, 1, 1)
        sobel_y_feat = self.sobel_y[0].repeat(channels, 1, 1, 1)

        grad_pred_x, grad_pred_y = F.conv2d(feat_map_pred, sobel_x_feat, padding=1, groups=channels), F.conv2d(feat_map_pred, sobel_y_feat, padding=1, groups=channels)
        with torch.no_grad():
            grad_gt_x, grad_gt_y = F.conv2d(feat_map_gt, sobel_x_feat, padding=1, groups=channels), F.conv2d(feat_map_gt, sobel_y_feat, padding=1, groups=channels)

        loss = l1_loss(grad_pred_x, grad_gt_x) + l1_loss(grad_pred_y, grad_gt_y)
        feedback_map_raw = (torch.abs(grad_pred_x - grad_gt_x) + torch.abs(grad_pred_y - grad_gt_y)).sum(dim=1, keepdim=True)
        return loss, F.interpolate(feedback_map_raw, size=pred.shape[-2:], mode='bilinear', align_corners=False)

    def _struct_ssim_loss(self, pred_b, gt_b):
        ssim_map = _ssim(pred_b, gt_b, create_window(self.opt.struct_ssim_window_size, 3).to(self.device), self.opt.struct_ssim_window_size, 3, reduction='none')
        feedback_map = 1.0 - ssim_map
        return torch.mean(feedback_map), feedback_map.detach().sum(dim=1, keepdim=True)

    def _wavelet_loss(self, pred_b, gt_b):
        pred_ll, pred_h = self.dwt(pred_b)
        with torch.no_grad():
            gt_ll, gt_h = self.dwt(gt_b)
        loss, diff_h = 0.0, []
        for i in range(len(pred_h)):
            loss += l1_loss(pred_h[i], gt_h[i])
            diff_h.append(torch.abs(pred_h[i] - gt_h[i]))
        with torch.no_grad():
            feedback_map = self.iwt((torch.zeros_like(pred_ll), diff_h))
            if feedback_map.shape[-2:] != pred_b.shape[-2:]:
                feedback_map = F.interpolate(feedback_map, size=pred_b.shape[-2:], mode='bilinear', align_corners=False)
        return loss, feedback_map.sum(dim=1, keepdim=True)


# ==============================================================================
#           协同引导模块 (负责“引导”与“修复”) - 已实现解耦模式
# ==============================================================================
class SynergyGuidance(nn.Module):
    def __init__(self, opt):
        super(SynergyGuidance, self).__init__()
        self.opt = opt
        self.device = 'cuda'

    def _normalize_map(self, m):
        if m.max() == m.min(): return m
        return (m - m.min()) / (m.max() - m.min() + 1e-8)
    
    def _apply_perceptual_weighting(self, struct_map, pred_im, gt_im, beta):
        """
        Applies perceptual weighting to the structural error map.
        M_Struct_Prime = M_Struct * (1 + beta * M_Perceptual)
        """
        with torch.no_grad():
            lum_pred = rgb_to_grayscale(pred_im).unsqueeze(0)
            lum_gt = rgb_to_grayscale(gt_im).unsqueeze(0)
            perceptual_map = torch.abs(lum_pred - lum_gt)
            return struct_map * (1.0 + beta * self._normalize_map(perceptual_map))

    def forward(self, feedback_map, pred_im, gt_im):
        W_l1 = torch.tensor(1.0, device=self.device)
        guided_ssim_loss = None

        if feedback_map is None or self.opt.synergy_mode == 'none':
            return W_l1, guided_ssim_loss

        processed_map = feedback_map.detach()
        mode = self.opt.synergy_mode


        if mode in ['v2_p_weighted', 'v4_fusion', 'v5_ultimate']:
            processed_map = self._apply_perceptual_weighting(
                processed_map, pred_im, gt_im, self.opt.feedback_p_weighting_beta
            )
        

        final_norm_map = self._normalize_map(processed_map)


        if mode == 'v5_ultimate':
            alpha_l1 = self.opt.alpha_l1_feedback
            gamma_l1 = self.opt.feedback_nonlinear_gamma
            
            map_for_l1 = torch.pow(final_norm_map, gamma_l1)
            W_l1 = 1.0 + alpha_l1 * map_for_l1
            
            alpha_ssim = self.opt.alpha_ssim_feedback
            ssim_weights = 1.0 + alpha_ssim * final_norm_map
            guided_ssim_loss = ssim_weighted(pred_im, gt_im, ssim_weights)

        else:
            alpha = self.opt.alpha_struct_feedback
            if alpha is not None and alpha > 0:
                # L1 Guidance (unified for older modes)
                if mode in ['v1_linear', 'v2_p_weighted', 'v4_fusion']:
                    W_l1 = 1.0 + alpha * final_norm_map
                elif mode == 'v2_nonlinear':
                    gamma = self.opt.feedback_nonlinear_gamma
                    W_l1 = 1.0 + alpha * torch.pow(final_norm_map, gamma)

                # SSIM Guidance (unified for older modes)
                if mode in ['v2_ssim_guided', 'v4_fusion']:
                    ssim_weights = 1.0 + alpha * final_norm_map
                    guided_ssim_loss = ssim_weighted(pred_im, gt_im, ssim_weights)

        return W_l1, guided_ssim_loss