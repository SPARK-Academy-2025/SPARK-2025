"""Defines common loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1 - dice
    

ALPHA = 1 # Could be 0.25?
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class NCCLoss(nn.Module):
    """Simple implementation for Normalized Cross Correlation that can be
    minimized with upper-bound of alpha and lower-bound of 0.
    """

    def __init__(self, alpha=1):
        super(NCCLoss, self).__init__()
        self.NCC = None  # -1(very dissimilar) to 1(very similar)
        self.alpha = alpha

    def forward(self, y, yp):
        EPSILON = 1e-6
        y_ = y - torch.mean(y)
        yp_ = yp - torch.mean(yp)
        self.NCC = torch.sum(
            y_ * yp_) / (((torch.sum(y_**2)) * torch.sum(yp_**2) + EPSILON)**0.5)
        error = self.alpha * (1 - self.NCC)
        return error