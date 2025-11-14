import torch
import numpy as np

def dice_coef_tensor(preds, targets, threshold=0.5, eps=1e-6):
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    targets_bin = (targets > 0.5).float()
    intersection = (preds_bin * targets_bin).sum(dim=(1,2,3))
    union = preds_bin.sum(dim=(1,2,3)) + targets_bin.sum(dim=(1,2,3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()

def iou_tensor(preds, targets, threshold=0.5, eps=1e-6):
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    targets_bin = (targets > 0.5).float()
    intersection = (preds_bin * targets_bin).sum(dim=(1,2,3))
    union = (preds_bin + targets_bin - preds_bin * targets_bin).sum(dim=(1,2,3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()
