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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

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

def _ssim(img1, img2, window, window_size, channel, size_average=True, reduction='mean'):
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
    else: #兼容旧的size_average
        return ssim_map.mean()

def ssim(img1, img2, window_size=11, reduction='mean'):
    # img1, img2: [C, H, W] -> [N, C, H, W]
    if len(img1.shape) == 3:
        img1, img2 = img1.unsqueeze(0), img2.unsqueeze(0)

    channel = img1.size(1)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, reduction=reduction).squeeze(0)


# ==============================================================================
#                       ‼️‼️ [核心新增] ‼️‼️
#         实现支持局部加权的 SA-SSIM (结构感知SSIM) 损失
# ==============================================================================
def ssim_weighted(img1, img2, weights, window_size=11):
    """
    计算加权的SSIM。
    Args:
        img1 (Tensor): 渲染图像, 形状 [C, H, W]
        img2 (Tensor): 真值图像, 形状 [C, H, W]
        weights (Tensor): 窗口权重图, 形状 [1, H, W]
    """
    # 1. 计算原始的、未平均的SSIM图
    # ssim() 函数返回 [H, W] 的图
    ssim_map = ssim(img1, img2, window_size=window_size, reduction='none')

    # 2. 对权重图进行平均池化，使其与窗口为中心的SSIM图对齐
    # weights: [1, H, W] -> [1, 1, H, W]
    # 使用与SSIM窗口相同的尺寸进行平均池化
    avg_pool = nn.AvgPool2d(window_size, stride=1, padding=window_size // 2)
    window_weights = avg_pool(weights.unsqueeze(0))

    # 3. 调整尺寸以匹配SSIM图 (通常不需要，但作为保险)
    if window_weights.shape[-2:] != ssim_map.shape[-2:]:
        window_weights = F.interpolate(window_weights, size=ssim_map.shape[-2:], mode='bilinear', align_corners=False)

    window_weights = window_weights.squeeze(0).squeeze(0) # -> [H, W]

    # 4. 计算加权平均
    weighted_ssim_score = torch.sum(ssim_map * window_weights) / torch.sum(window_weights)

    return weighted_ssim_score