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
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import random

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
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

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def loss_photometric(image, gt_image, opt, valid=None):
    Ll1 =  l1_loss_mask(image, gt_image, mask=valid)
    loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=valid)))
    return loss


def margin_l2_loss(network_output, gt, mask_patches, margin, return_mask=False):
    network_output = network_output[mask_patches]
    gt = gt[mask_patches]
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask


def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2 * torch.std(input.reshape(-1)))


def patchify(input, patch_size):
    # 如果输入类型是 bool，则转换为 float
    if input.dtype == torch.bool:
        input = input.float()
    # 如果输入是 3D (C, H, W)，增加 batch 维度
    if input.dim() == 3:
        input = input.unsqueeze(0)
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size)
    patches = patches.permute(0, 2, 1).contiguous().view(-1, input.size(1) * patch_size * patch_size)
    return patches


def patch_norm_mse_loss(input, target, fore_mask, patch_size, margin=0.2, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    mask_patches = patchify(fore_mask, patch_size).sum(dim=1) < (patch_size * patch_size / 3)
    return margin_l2_loss(input_patches, target_patches, mask_patches, margin, return_mask)


def ranking_loss(input, target, patch_size, margin=1e-4):
    input_patches = patchify(input, patch_size)
    target_patches = patchify(target, patch_size)

    rand_idxes = random.sample(list(range(input_patches.shape[1])), 6)

    input_pixels = input_patches[:, rand_idxes].reshape(-1, 2)
    target_pixels = target_patches[:, rand_idxes].reshape(-1, 2)

    g = target_pixels[:, 0] - target_pixels[:, 1]
    t = input_pixels[:, 0] - input_pixels[:, 1]

    t = torch.where(g < 0, t, -t)

    t = t + margin

    l = torch.mean(t[t > 0])

    return l
