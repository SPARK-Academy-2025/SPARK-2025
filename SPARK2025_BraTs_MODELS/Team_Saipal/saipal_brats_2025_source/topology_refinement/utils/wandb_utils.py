import os
import json
import torch
import numpy as np
import wandb
from utils.utils import prepare_image_for_logging

def image_tuple(input, target, output):
    """
    Provides Input, Target and Predicted Image Tuples for Logging
    """
    input_img = torch.argmax(input, dim=1)[0].cpu().numpy()
    target_img = torch.argmax(target, dim=1)[0].cpu().numpy()  # target shape: [C, D, H, W]
    output_img = torch.argmax(output, dim=1)[0].detach().cpu().numpy()
    mid_slice = input_img.shape[2] // 2 # Depth slice index (should be 2 for depth)

    input_img = np.fliplr(np.rot90(input_img, k=1))
    target_img = np.fliplr(np.rot90(target_img, k=1))
    output_img = np.fliplr(np.rot90(output_img, k=1))
        
    return input_img, target_img, output_img, mid_slice

def log_train_image(model, input, target, sample_no, slice_id=None):
    """
    Log train dataset image for a fixed sample

    Args:
        model (nn.Module): The model to use for inference.
        input (torch.Tensor): Input tensor of shape [B, C, D, H, W].
        target (torch.Tensor): Target tensor of shape [B, C, D, H, W].
        sample_no (int): Sample number for logging.
        slice_id (int, optional): Specific slice index to log and slice index should start from 1 not 0. If None, uses the middle slice.
    Returns:
        dict: Dictionary containing wandb images for input, target, and output.
    """
    output = model(input)
    input_img, target_img, output_img, mid_slice = image_tuple(input, target, output)
    mid_slice = (slice_id - 1) if slice_id is not None else mid_slice

    if wandb.run is not None:
        return {
            "train/input": wandb.Image(prepare_image_for_logging(input_img[:, :, mid_slice]), caption=f"Train Input (Sample {sample_no})"),
            "train/target": wandb.Image(prepare_image_for_logging(target_img[:, :, mid_slice]), caption=f"Train Mask (Sample {sample_no})"),
            "train/output": wandb.Image(prepare_image_for_logging(output_img[:, :, mid_slice]), caption=f"Train Predicted (Sample {sample_no})"),
        }
    
    return {}

def log_test_image(model, input, target, sample_no, slice_id=None):
    """
    Log test dataset image for a fixed sample
    """
    output = model(input)
    input_img, target_img, output_img, mid_slice = image_tuple(input, target, output)
    mid_slice = (slice_id - 1) if slice_id is not None else mid_slice

    if wandb.run is not None:
        return {
            "val/input": wandb.Image(prepare_image_for_logging(input_img[:, :, mid_slice]), caption=f"Val Input (Sample {sample_no})"),
            "val/target": wandb.Image(prepare_image_for_logging(target_img[:, :, mid_slice]), caption=f"Val Mask (Sample {sample_no})"),
            "val/output": wandb.Image(prepare_image_for_logging(output_img[:, :, mid_slice]), caption=f"Val Predicted (Sample {sample_no})"),
        }
    
    return {}

def train_image_sample(model, input, target, sample_no, slice_id=None):
    output = model(input) if model is not None else None
    input_img, target_img, output_img, mid_slice = image_tuple(input, target, output)
    mid_slice = slice_id - 1 if slice_id is not None else mid_slice

    return prepare_image_for_logging(input_img[:, :, mid_slice]), prepare_image_for_logging(target_img[:, :, mid_slice]), prepare_image_for_logging(output_img[:, :, mid_slice]), mid_slice

def init_wandb(project, run_name=None, config=None, resume=False, id_file="wandb_run_id.txt"):
    """
    Initializes wandb and handles resumption logic via a run_id file.

    Args:
        project (str): W&B project name.
        run_name (str): Name of the W&B run.
        config (dict): W&B config dictionary.
        resume (bool): Whether to resume from a previous run.
        id_file (str): Path to the file where wandb run ID is stored.
    """
    wandb_id = None

    if resume and os.path.exists(id_file):
        with open(id_file, "r") as f:
            data = json.load(f)
            wandb_id = data['id']
            run_name = data['name']
        print(f"Resuming W&B run with ID: {wandb_id}")
        run = wandb.init(project=project, id=wandb_id, resume="must", name=run_name)
    else:
        run = wandb.init(project=project, config=config, name=run_name)
        data = {'id': run.id, 'name': run.name}
        with open(id_file, "w") as f:
            json.dump(data, f)
    print(f"W&B run initialized with ID: {wandb.run.id}")