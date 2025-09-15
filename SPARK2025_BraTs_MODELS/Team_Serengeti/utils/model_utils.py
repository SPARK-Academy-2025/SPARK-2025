"""This module contains utility functions for training models."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..datasets import brats_dataset
from .general_utils import seg_to_one_hot_channels, disjoint_to_overlapping

def load_or_initialize_training(model, optimizer, latest_ckpt_path, train_with_val=False):
    """Loads training checkpoint if it exists, or initializes training from scratch.

    Args:
        model: The PyTorch model to be trained.
        optimizer: The optimizer used for training.
        latest_ckpt_path: The path to the latest model checkpoint.
        train_with_val: If True, also returns best saved validation loss and dice. Defaults to False.

    Returns:
        The starting epoch number.
        If 'train_with_val' is True, also returns best saved validation loss and dice.
    """

    if not os.path.exists(latest_ckpt_path):
        epoch_start = 1
        if train_with_val:
            best_vloss = float('inf')
            best_dice = 0
        print('No training checkpoint found. Will start training from scratch.')
    else:
        print('Training checkpoint found. Loading checkpoint...')
        checkpoint = torch.load(latest_ckpt_path,  weights_only=False)
        epoch_start = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_sd'])
        optimizer.load_state_dict(checkpoint['optim_sd'])
        if train_with_val:
            best_vloss = checkpoint['vloss']
            best_dice = checkpoint['dice']
        print(f'Checkpoint loaded. Will continue training from epoch {epoch_start}.')

    if train_with_val:
        return epoch_start, best_vloss, best_dice
    return epoch_start

def make_dataloader(data_dir, shuffle, mode, augment, batch_size=1, max_subjects=None):
    dataset = brats_dataset.BratsDataset(data_dir, mode=mode, augment=augment, max_subjects=max_subjects)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True)
    return dataloader

def exp_decay_learning_rate(optimizer, epoch, init_lr, decay_rate):
    """Exponentially decays learning rate of optimizer at given epoch."""
    lr = init_lr * (decay_rate ** (epoch-1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def compute_loss(output, seg, loss_functs, loss_weights):
    """Computes weighted loss between model output and ground truth, summed across each region."""
    loss = 0.
    for n, loss_function in enumerate(loss_functs):      
        temp = 0
        for i in range(3):
            temp += loss_function(output[:,i:i+1].cuda(), seg[:,i:i+1].cuda())

        loss += temp * loss_weights[n]
    return loss

def train_one_epoch(model, optimizer, train_loader, loss_functions, loss_weights, training_regions):
    """Performs one training loop of model according to given optimizer, loss functions and associated weights.

    Args:
        model: The PyTorch model to be trained.
        optimizer: The optimizer used for training.
        train_loader: The dataloader for training data.
        loss_functions: List of loss functions.
        loss_weights: List of associated weightings for each loss function.
        training_regions: String specifying whether 'disjoint' or 'overlapping' regions will be used for training.

    Returns:
        The average training loss over the epoch.
    """
    losses_over_epoch = []
    for _, imgs, seg in train_loader:
        model.train()

        # Move data to GPU.
        imgs = [img.cuda() for img in imgs] # img is B1HWD
        seg = seg.cuda()

        # Split segmentation into 3 channels.
        seg = seg_to_one_hot_channels(seg)
        # seg is B3HWD - each channel is one-hot encoding of a disjoint region

        if training_regions == 'overlapping':
            seg = disjoint_to_overlapping(seg)
            # seg is B3HWD - each channel is one-hot encoding of an overlapping region

        x_in = torch.cat(imgs, dim=1) # x_in is B4HWD
        outputs = model(x_in)
        
        # Handle both dictionary output (deep supervision) and single output cases
        if isinstance(outputs, dict):
            # Compute loss for each output and combine them
            loss = 0
            for output in outputs.values():
                output = output.float()
                loss += compute_loss(output, seg, loss_functions, loss_weights)
            # Normalize by number of outputs
            loss = loss / len(outputs)
        else:
            # Original single output case
            outputs = outputs.float()
            loss = compute_loss(outputs, seg, loss_functions, loss_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_over_epoch.append(loss.detach().cpu())

    # Compute loss from the epoch.
    average_epoch_loss = np.mean(losses_over_epoch)
    return average_epoch_loss



def freeze_layers(model, frozen_layers):
    """Freezes specified model layers. Afterwards parameters in these layers will not be updated when training.

    Args:
        model: The model to be trained.
        frozen_layers: List of strings specifying model layers.
    """

    for name, param in model.named_parameters():
        needs_freezing = False
        for layer in frozen_layers:
            if layer in name:
                needs_freezing = True
                break
        if needs_freezing:
            print(f'Freezing parameter {name}.')
            param.requires_grad = False

def check_frozen(model, frozen_layers):
    """Iterates through model layers and checks whether specified layers are frozen.

    Args:
        model: The model to be trained.
        frozen_layers: List of strings specifying model layers.
    """
    for name, param in model.named_parameters():
        needs_freezing = False
        for layer in frozen_layers:
            if layer in name:
                needs_freezing = True
                break
        if needs_freezing:
            if param.requires_grad:
                print(f'Warning! Param {name} should not require grad but does.')
                break
            else:
                print(f'Parameter {name} is frozen.')

# Example parts of unet_3d model to freeze
# 'encoder': ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Conv7'],
# 'decoder': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4', 'Up3', 'Up_conv3', 'Conv_1x13', 'Up2', 'Up_conv2', 'Conv_1x12', 'Up1', 'Up_conv1', 'Conv_1x11'],
# 'middle' : ['Conv5', 'Conv6', 'Conv7', 'Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4'],
# 'none' : [],
# 'deep_decoder': ['Up6', 'Up_conv6', 'Up5', 'Up_conv5', 'Up4', 'Up_conv4']