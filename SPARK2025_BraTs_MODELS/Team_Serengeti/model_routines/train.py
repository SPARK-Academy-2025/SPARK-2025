import os
import numpy as np
import torch 
from torch import optim
import csv
import torch.nn as nn

from monai.losses import DiceLoss, FocalLoss, DiceCELoss

from ..utils.model_utils import load_or_initialize_training, make_dataloader, exp_decay_learning_rate, train_one_epoch
    
def train(data_dir, model, loss_functions, loss_weights, init_lr, max_epoch, training_regions='overlapping', out_dir=None, decay_rate=0.995, backup_interval=10, batch_size=2, scheduler_patience=5):
    """Runs basic training routine.

    Args:
        data_dir: Directory of training data.
        model: The PyTorch model to be trained.
        loss_functions: List of loss functions to be used for training.
        loss_weights: List of weights corresponding to each loss function.
        init_lr: Initial value of learning rate.
        max_epoch: Maximum number of epochs to train for.
        training_regions: Whether training on 'disjoint' or 'overlapping' regions. Defaults to 'overlapping'.
        out_dir: The directory to save model checkpoints and loss values. Defaults to None.
        decay_rate: Rate at which to decay the learning rate. Defaults to 0.995.
        backup_interval: How often to save a backup checkpoint. Defaults to 10.
        batch_size: Batch size of dataloader. Defaults to 1.
        scheduler_patience: Patience for ReduceLROnPlateau scheduler. Defaults to 5.
    """
    # Set up directories and paths.
    if out_dir is None:
        out_dir = os.getcwd()
    latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')
    training_loss_path = os.path.join(out_dir, 'training_loss.csv')
    backup_ckpts_dir = os.path.join(out_dir, 'backup_ckpts')
    if not os.path.exists(backup_ckpts_dir):
        os.makedirs(backup_ckpts_dir)
        os.system(f'chmod a+rwx {backup_ckpts_dir}')

    print("---------------------------------------------------")
    print(f"TRAINING SUMMARY")
    print(f"Data directory: {data_dir}")
    print(f"Model: {model}")
    print(f"Loss functions: {loss_functions}") 
    print(f"Loss weights: {loss_weights}")
    print(f"Initial learning rate: {init_lr}")
    print(f"Max epochs: {max_epoch}")
    print(f"Training regions: {training_regions}")
    print(f"Out directory: {out_dir}")
    print(f"Decay rate: {decay_rate}")
    print(f"Backup interval: {backup_interval}")
    print(f"Batch size: {batch_size}")
    print(f"Scheduler patience: {scheduler_patience}")
    print("---------------------------------------------------")

    # Enhanced optimizer: AdamW with proper weight decay
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, weight_decay=1e-5, amsgrad=True)
    
    # Learning rate scheduler: ReduceLROnPlateau for adaptive learning rate adjustment
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=scheduler_patience, verbose=True)

    # Check if training for first time or continuing from a saved checkpoint.
    epoch_start = load_or_initialize_training(model, optimizer, latest_ckpt_path)

    train_loader = make_dataloader(data_dir, shuffle=True, mode='train', augment=True, 
                                   batch_size=batch_size, max_subjects=60)

    print('Training starts.')
    for epoch in range(epoch_start, max_epoch+1):
        print(f'Starting epoch {epoch}...')

        # Note: We now use ReduceLROnPlateau instead of exponential decay
        # The scheduler will adjust LR based on loss plateau
        
        average_epoch_loss = train_one_epoch(model, optimizer, train_loader, loss_functions, loss_weights, training_regions)

        # Step the scheduler with the current loss
        scheduler.step(average_epoch_loss)

        # Save and report loss from the epoch.
        save_tloss_csv(training_loss_path, epoch, average_epoch_loss)
        print(f'Epoch {epoch} completed. Average loss = {average_epoch_loss:.4f}.')
        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

        print('Saving model checkpoint...')
        checkpoint = {
            'epoch': epoch,
            'model_sd': model.state_dict(),
            'optim_sd': optimizer.state_dict(),
            'scheduler_sd': scheduler.state_dict(),
            'model': model,
            'loss_functions': loss_functions,
            'loss_weights': loss_weights,
            'init_lr': init_lr,
            'training_regions': training_regions,
            'decay_rate': decay_rate,
            'scheduler_patience': scheduler_patience
        }
        torch.save(checkpoint, latest_ckpt_path)
        if epoch % backup_interval == 0:
            torch.save(checkpoint, os.path.join(backup_ckpts_dir, f'epoch{epoch}.pth.tar'))
        print('Checkpoint saved successfully.')

    
def save_tloss_csv(pathname, epoch, tloss):
    with open(pathname, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            writer.writerow(['Epoch', 'Training Loss'])
        writer.writerow([epoch, tloss])

if __name__ == '__main__':

    from ..models import unet3d
    import torch.nn as nn

    data_dir = '../ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2'
    model = U_Net3d()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Enhanced loss functions: DiceCE (combines Dice and CrossEntropy) + FocalLoss
    loss_functions = [
        DiceCELoss(include_background=False, softmax=True, lambda_dice=0.5, lambda_ce=0.5),
        FocalLoss(gamma=2.0, weight=torch.tensor([1.0, 2.0, 3.0], device=device))  # Higher weight for tumor classes
    ]
    loss_weights = [0.7, 0.3]  # Favor DiceCE over Focal

    lr = 3e-4
    max_epoch = 150
    #out_dir = '/home/mailab/Documents/brats2023_updated/Result'
    out_dir = '../Result'

    train(data_dir, model, loss_functions, loss_weights, lr, max_epoch, out_dir=out_dir)