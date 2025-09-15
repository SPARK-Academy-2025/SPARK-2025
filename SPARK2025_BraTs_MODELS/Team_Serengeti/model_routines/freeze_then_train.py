import os
import numpy as np
import torch
from torch import optim
import csv

from ..utils.model_utils import load_or_initialize_training, freeze_layers, make_dataloader, check_frozen, exp_decay_learning_rate, compute_loss, train_one_epoch
from ..utils.general_utils import seg_to_one_hot_channels, disjoint_to_overlapping

def freeze_then_continue_training(data_dir, prev_ckpt_path, max_epoch, frozen_layers, out_dir=None, backup_interval=10, batch_size=1):
    """Continues training of a model (on a potentially new training dataset) after freezing specific layers of the model.

    Args:
        data_dir: Directory of training data.
        prev_ckpt_path: Path of previously trained model.
        max_epoch: Maximum number of epochs to train for.
        frozen_layers: List of model layers to be frozen.
        out_dir: The directory to save model checkpoints and loss values. Defaults to None.
        backup_interval: How often to save a backup checkpoint. Defaults to 10.
        batch_size: Batch size of dataloader. Defaults to 1.
    """

    # Set up directories and paths.
    if out_dir is None:
        out_dir = os.getcwd()
    latest_ckpt_path = os.path.join(out_dir, 'latest_ckpt.pth.tar')
    training_loss_path = os.path.join(out_dir, 'training_loss.csv')
    backup_ckpts_dir = os.path.join(out_dir, 'new_backup_ckpts')
    if not os.path.exists(backup_ckpts_dir):
        os.makedirs(backup_ckpts_dir)
        os.system(f'chmod a+rwx {backup_ckpts_dir}')

    print(f"Loading model from {prev_ckpt_path}...")
    checkpoint = torch.load(prev_ckpt_path)

    model = checkpoint['model']
    loss_functions = checkpoint['loss_functions']
    loss_weights = checkpoint['loss_weights']
    training_regions = checkpoint['training_regions']
    init_lr = checkpoint['init_lr']
    decay_rate = checkpoint['decay_rate']

    epoch = checkpoint['epoch']
    model_sd = checkpoint['model_sd']

    model.load_state_dict(model_sd)

    print("---------------------------------------------------")
    print(f"PREVIOUS TRAINING SUMMARY")
    print(f"Model: {model}")
    print(f"Loss functions: {loss_functions}") 
    print(f"Loss weights: {loss_weights}")
    print(f"Training regions: {training_regions}")
    print(f"Epochs trained: {epoch}")
    print("---------------------------------------------------")
    print("CONTINUING TRAINING SUMMARY")
    print(f"Data directory: {data_dir}")
    print(f"Trained model checkpoint path: {prev_ckpt_path}")
    print(f"Initial learning rate: {init_lr}")
    print(f"Max epochs: {max_epoch}")
    print(f"Out directory: {out_dir}")
    print(f"Decay rate: {decay_rate}")
    print(f"Backup interval: {backup_interval}")
    print(f"Batch size: {batch_size}")
    print("---------------------------------------------------")

    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=0, amsgrad=True)

    # Check if continuing training for first time or continuing from a saved checkpoint.
    epoch_start = load_or_initialize_training(model, optimizer, latest_ckpt_path)

    freeze_layers(model, frozen_layers)

    train_loader = make_dataloader(data_dir, shuffle=True, mode='train')

    print('Training starts.')
    for epoch in range(epoch_start, max_epoch+1):
        print(f'Starting epoch {epoch}...')

        # Check model is frozen appropriately at start.
        if epoch == epoch_start:
            check_frozen(model, frozen_layers)

        exp_decay_learning_rate(optimizer, epoch, init_lr, decay_rate)

        average_epoch_loss = train_one_epoch(model, optimizer, train_loader, loss_functions, loss_weights, training_regions)

        # Save and report loss from the epoch.
        save_tloss_csv(training_loss_path, epoch, average_epoch_loss)
        print(f'Epoch {epoch} completed. Average loss = {average_epoch_loss:.4f}.')

        print('Saving model checkpoint...')
        checkpoint = {
            'epoch': epoch,
            'model_sd': model.state_dict(),
            'optim_sd': optimizer.state_dict(),
            'model': model,
            'loss_functions': loss_functions,
            'loss_weights': loss_weights,
            'init_lr': init_lr,
            'training_regions': training_regions,
            'decay_rate': decay_rate
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

    data_dir = '/mmfs1/home/ehoney22/debug_data/train'
    prev_ckpt_path = '/mmfs1/home/ehoney22/debug/backup_ckpts/epoch20.pth.tar'
    max_epoch = 20
    frozen_layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'Conv6', 'Conv7']
    out_dir = '/mmfs1/home/ehoney22/debug/freeze_then_continue_training'

    freeze_then_continue_training(data_dir, prev_ckpt_path, max_epoch, frozen_layers, out_dir=out_dir)