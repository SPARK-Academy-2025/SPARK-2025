"""This module contains code for plotting MRI images alongside ground truth and predicted segmentations."""

import numpy as np
import matplotlib.pyplot as plt

def max_slice(seg):
    """Finds the slice (in the z-direction) of a segmentation with the most tumor voxels."""
    one_hot_encoded = np.where(seg != 0, 1, 0)

    slice_sums = np.sum(one_hot_encoded, axis=(0,1))
    max_index = np.argmax(slice_sums)

    return max_index

def plot_slices(images=None, seg=None, pred=None, nslice=None):
    """For a given subject, plots same slice of MRI images, ground truth and predicted segmentations, if provided.

    Args:
        images: List (one for each of the four modalities) of MRI image tensors of shape HWD. Defaults to None.
        seg: Ground truth segmentation tensor of shape HWD. Defaults to None.
        pred: Predicted segmentation tensor of shape HWD. Defaults to None.
        nslice: Slice number of images and segs to plot. Defaults to None.

    Returns:
        Matplotlib figure containing the plot.
    """
    n_rows = 0
    if images is not None:
        n_rows += 1
    if seg is not None:
        n_rows += 1
    if pred is not None:
        n_rows += 1

    if nslice is None:
        if seg is not None:
            nslice = max_slice(seg)
        elif pred is not None:
            nslice = max_slice(pred)
        else:
            nslice=64

    fig, axes = plt.subplots(n_rows, 4, figsize=(15, 5*n_rows))

    row = 0
    if images is not None:
        for i, suffix in enumerate(['t1c', 't1n', 't2f', 't2w']):
            axes[row, i].imshow(images[i][:, :, nslice], cmap='gray')
            axes[row, i].set_title(f'{suffix}')
            axes[row, i].axis('off')
        row += 1

    if seg is not None:
        seg_slice = seg[:, :, nslice]
        for i, region in enumerate(['NCR', 'ED', 'ET']):
            axes[row, i].imshow(np.where(seg_slice == i+1, i+1, 0), cmap='viridis', vmin=0, vmax=3)
            axes[row, i].set_title(f'{region} ground truth')
            axes[row, i].axis('off')
        axes[row, 3].imshow(seg_slice, cmap='viridis', vmin=0, vmax=3)
        axes[row, 3].set_title('All regions ground truth')
        axes[row, 3].axis('off')
        row += 1

    if pred is not None:
        pred_slice = pred[:, :, nslice]
        for i, region in enumerate(['NCR', 'ED', 'ET']):
            axes[row, i].imshow(np.where(pred_slice == i+1, i+1, 0), cmap='viridis', vmin=0, vmax=3)
            axes[row, i].set_title(f'{region} prediction')
            axes[row, i].axis('off')
        axes[row, 3].imshow(pred_slice, cmap='viridis', vmin=0, vmax=3)
        axes[row, 3].set_title('All regions prediction')
        axes[row, 3].axis('off')

    plt.tight_layout()
    return fig