import os
import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric, DiceMetric

from ..utils.model_utils import make_dataloader, compute_loss
from ..utils.general_utils import seg_to_one_hot_channels, disjoint_to_overlapping, probs_to_preds, one_hot_channels_to_three_labels
from ..processing.plot import plot_slices

def validate(data_dir, ckpt_path, eval_regions='overlapping', out_dir=None, make_plots=False, batch_size=1):
    """Routine to validate a trained model on validation data. Optionally plots predictions against ground truth segmentations.
    
    Args:
        data_dir: Directory of validation data.
        ckpt_path: Path of trained model.
        eval_regions: Whether to evaluate on 'disjoint' or 'overlapping' regions. Defaults to 'overlapping'.
        out_dir: Directory in which to save plots. Defaults to None.
        make_plots: Whether to produce plots of predictions and ground truth segmentations. Defaults to False.
        batch_size: Batch size of dataloader. Defaults to 1.
    """

    # Set up directories.
    if out_dir is None:
        out_dir = os.getcwd()

    if make_plots:
        plots_dir = os.path.join(out_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            os.system(f'chmod a+rwx {plots_dir}')

    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path)

    model = checkpoint['model']
    loss_functions = checkpoint['loss_functions']
    loss_weights = checkpoint['loss_weights']
    training_regions = checkpoint['training_regions']

    epoch = checkpoint['epoch']
    model_sd = checkpoint['model_sd']

    model.load_state_dict(model_sd)

    print('Model loaded.')

    print("---------------------------------------------------")
    print(f"TRAINING SUMMARY")
    print(f"Model: {model}")
    print(f"Loss functions: {loss_functions}") 
    print(f"Loss weights: {loss_weights}")
    print(f"Training regions: {training_regions}")
    print(f"Epochs trained: {epoch}")
    print("---------------------------------------------------")
    print("VALIDATION SUMMARY")
    print(f"Data directory: {data_dir}")
    print(f"Trained model checkpoint path: {ckpt_path}")
    print(f"Evaluation regions: {eval_regions}")
    print(f"Out directory: {out_dir}")
    print(f"Make plots: {make_plots}")
    print(f"Batch size: {batch_size}")
    print("---------------------------------------------------")

    val_loader = make_dataloader(data_dir, shuffle=False, mode='train', batch_size=batch_size)

    val_loss_vals = []

    # Recommend use MONAI metrics set-up for different metrics (Cumulative Iterative)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean_batch")

    print('Validation starts.')
    with torch.no_grad():
        for subject_names, imgs, seg in val_loader:

            model.eval()

            # Move data to GPU.
            imgs = [img.cuda() for img in imgs] # img is B1HWD
            seg = seg.cuda()

            # Split segmentation into 3 channels.
            seg = seg_to_one_hot_channels(seg) # seg is B3HWD

            if training_regions == 'overlapping':
                seg_train = disjoint_to_overlapping(seg)
                # seg_train is B3HWD - each channel is one-hot encoding of an overlapping region
            elif training_regions == 'disjoint':
                seg_train = seg
                # seg_train is B3HWD - each channel is one-hot encoding of a disjoint region

            x_in = torch.cat(imgs, dim=1) # x_in is B4HWD
            output = model(x_in)
            output = output.float()

            # Compute weighted loss, summed across each training region.
            val_loss = compute_loss(output, seg_train, loss_functions, loss_weights)
            val_loss_vals.append(val_loss.detach().cpu())

            preds = probs_to_preds(output, training_regions)

            eval_region_names = []
            if eval_regions == 'overlapping':
                eval_region_names = ['WT', 'TC', 'ET']
                # Convert seg and pred to 3 channels corresponding to overlapping regions
                seg_eval = disjoint_to_overlapping(seg)
                preds_eval = disjoint_to_overlapping(preds)
                
            elif eval_regions == 'disjoint':
                eval_region_names = ['NCR', 'ED', 'ET']
                # Convert seg and pred to 3 channels corresponding to disjoint regions
                seg_eval = seg
                preds_eval = preds

            # seg_eval is B3HWD
            # preds_eval is B3HWD

            # Compute metrics between seg_eval and preds_eval.
            dice_metric(y_pred = preds_eval, y=seg_eval)
            hd_metric(y_pred = preds_eval, y=seg_eval)

            if make_plots:
                # Make plots for each subject in batch.
                for i, subject_name in enumerate(subject_names):

                    batch_imgs = [img[i, 0].cpu().detach() for img in imgs]
                    seg3 = one_hot_channels_to_three_labels(seg[i].cpu().detach())
                    pred3 = one_hot_channels_to_three_labels(preds[i])

                    fig = plot_slices(batch_imgs, seg3, pred3)
                    fig.savefig(os.path.join(plots_dir, subject_name))

    # Compute and report validation loss.
    average_val_loss = np.mean(val_loss_vals)
    print(f'Validation completed. Average validation loss = {average_val_loss}')

    # Aggregate and report the metrics.
    dice_metric_batch = dice_metric.aggregate()
    for i, eval_region in enumerate(eval_region_names):
        print(f'Dice Score {eval_region} = {dice_metric_batch[i].item()}')
    hd_metric_batch = hd_metric.aggregate()
    for i, eval_region in enumerate(eval_region_names):
        print(f'HD95 {eval_region} = {hd_metric_batch[i].item()}')

if __name__ == '__main__':

    data_dir = '/mmfs1/home/ehoney22/debug_data/train'
    ckpt_path = '/mmfs1/home/ehoney22/debug/backup_ckpts/epoch20.pth.tar'
    out_dir = '/mmfs1/home/ehoney22/debug'

    validate(data_dir, ckpt_path, out_dir=out_dir)