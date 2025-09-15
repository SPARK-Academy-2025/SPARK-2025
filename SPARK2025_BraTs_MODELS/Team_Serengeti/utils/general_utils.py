
import torch
import numpy as np
import nibabel as nib
import os
from ..processing.preprocess import undo_center_crop

def seg_to_one_hot_channels(seg):
    """Converts segmentation to 3 channels, each a one-hot encoding of a tumour region label.

    Args:
        seg: Tensor of shape B1HWD, where each entry is a voxel label.

    Returns:
        Tensor of shape B3HWD, where each channel is one-hot encoding of a disjoint region.
    """
    B, _, H, W, D = seg.shape
    seg3 = torch.zeros((B, 3, H, W, D), device=seg.device)
    
    # Squeeze the channel dimension for comparison
    seg = seg.squeeze(1)  # Now shape [B, H, W, D]
    
    for channel_value in [1, 2, 3]:
        # Create mask and add channel dimension back
        mask = (seg == channel_value).float().unsqueeze(1)  # Shape [B, 1, H, W, D]
        seg3[:, channel_value-1:channel_value, :, :, :] = mask
    
    return seg3

def disjoint_to_overlapping(seg_disjoint):
    """Converts tensor representing one-hot encoding of disjoint regions to that of overlapping ones.

    Args:
        seg_disjoint: Tensor of shape B3HWD, where each channel is one-hot encoding of a disjoint region. 

    Returns:
        Tensor of shape B3HWD, where each channel is one-hot encoding of an overlapping region.
    """
    seg_overlapping = torch.zeros_like(seg_disjoint)
    seg_overlapping[:,0] = seg_disjoint[:, 0] + seg_disjoint[:, 1] + seg_disjoint[:, 2] #WHOLE TUMOR
    seg_overlapping[:,1] = seg_disjoint[:, 0] + seg_disjoint[:, 2] #TUMOR CORE
    seg_overlapping[:,2] = seg_disjoint[:, 2] #ENHANCING TUMOR
    return seg_overlapping

def overlapping_probs_to_preds(output, t1=0.45, t2=0.4, t3=0.45):
    """Converts output of model trained on overlapping regions to one-hot encodings of disjoint regions.

    Args:
        output: Tensor of shape B3HWD. Output of model, representing probabilties each voxel belongs to each overlapping region.
        t1: Threshold for being in whole tumor (WT). Defaults to 0.45.
        t2: Threshold for being in tumor core (TC). Defaults to 0.4.
        t3: Threshold for being in enhancing tumor (ET). Defaults to 0.45.

    Returns:
        Tensor of shape B3HWD, where each channel is one-hot encoding of a disjoint region.
    """
    output = output.cpu().detach()
    c1, c2, c3 = output[:, 0] > t1, output[:, 1] > t2, output[:, 2] > t3
    preds = (c1 > 0).to(torch.uint8) # NCR
    preds[(c2 == False) * (c1 == True)] = 2 # ED
    preds[(c3 == True) * (c1 == True)] = 3 # ET
    output_plot = torch.zeros_like(output)
    output_plot[:, 0] = (preds == 1).to(torch.uint8) #NCR
    output_plot[:, 1] = (preds == 2).to(torch.uint8) #ED
    output_plot[:, 2] = (preds == 3).to(torch.uint8) #ET
    output_plot = output_plot.to(torch.uint8)
    return output_plot

def disjoint_probs_to_preds(output, t=0.5):
    """Converts output of model trained on disjoint regions to one-hot encodings of disjoint regions.

    Args:
        output: Tensor of shape B3HWD. Output of model, representing probabilties each voxel belongs to each disjoint region.
        t: Threshold value. If channel probability for a voxel is the maximum across all channels AND greater than this threshold, channel value will be encoded as 1, otherwise 0. Defaults to 0.5.

    Returns:
        Tensor of shape B3HWD, where each channel is one-hot encoding of a disjoint region.
    """
    output = output.cpu().detach()
    c1, c2, c3 = output[:, 0], output[:, 1], output[:, 2]
    max_label = torch.max(torch.max(c1, c2), c3)
    preds = torch.zeros_like(output)
    preds[:, 0] = torch.where(c1 < max_label, torch.tensor(0), max_label)
    preds[:, 1] = torch.where(c2 < max_label, torch.tensor(0), max_label)
    preds[:, 2] = torch.where(c3 < max_label, torch.tensor(0), max_label)
    output_plot = torch.zeros_like(output)
    for c in range(0, 3):
        output_plot[:, c] = torch.where(preds[:, c] > t, torch.tensor(1.), torch.tensor(0.))
    output_plot = output_plot.to(torch.uint8)
    return output_plot

def probs_to_preds(output, training_regions):
    """Converts tensor of voxel probabilities to tensor of disjoint region labels.

    Args:
        output: Tensor of shape B3HWD. Output of model, representing probabilties each voxel belongs to a region.
        training_regions: Whether probabilities relate to overlapping or disjoint regions.

    Returns:
        Tensor of shape B3HWD, where each channel is one-hot encoding of a disjoint region.
    """
    if training_regions == 'overlapping':
        preds = overlapping_probs_to_preds(output)
    elif training_regions == 'disjoint':
        preds = disjoint_probs_to_preds(output)

    return preds

def fetch_affine_header(subject_name, data_dir):
    """Finds affine and header of a modality nifti for given subject.

    Args:
        subject_name: Name of given subject. Will also be name of folder containing MRI niftis.
        data_dir: Parent directory of subject data folder.

    Returns:
        The affine and header objects from a modality nifti of the subject.
    """

    modality_nifti_filename = f'{subject_name}-t1c.nii'
    modality_nifti_path = os.path.join(data_dir, subject_name, modality_nifti_filename)
    nifti = nib.load(modality_nifti_path)
    
    return nifti.affine, nifti.header

def one_hot_channels_to_three_labels(pred):
    """Converts tensor of one-hot encodings of disjoint regions to be single channel. Each voxel is assigned a single disjoint region label.

    Args:
        pred: Array-like of shape 3HWD, where channels are one-hot encodings of disjoint regions.

    Returns:
        Array-like of shape HWD, associating to each voxel a single disjoint region label.
    """
    return pred[0] + pred[1]*2 + pred[2]*3

def save_pred_as_nifti(pred, save_dir, data_dir, subject_name, postprocess_function=None):
    """Saves predicted segmentation as nifti file with affine and header objects matching its MRI niftis.

    Args:
        pred: Tensor of shape HWD, associating to each voxel a single disjoint region label.
        save_dir: Directory in which to save the predicted segmentation nifti.
        data_dir: Parent directory of subject data folder.
        subject_name: Name of given subject. Will also be name of folder containing MRI niftis.
        postprocess_function: If provided, performs this postprocessing on the prediction. Defaults to None.
    """
    # Convert back from 3 one-hot encoded channels to 1 channel with 3 tumour region labels
    pred = np.array(pred)
    pred_for_nifti = one_hot_channels_to_three_labels(pred)
    pred_for_nifti = np.squeeze(pred_for_nifti)
    pred_for_nifti = undo_center_crop(pred_for_nifti)
    pred_for_nifti = pred_for_nifti.astype(np.uint8)

    if postprocess_function:
        pred_for_nifti = postprocess_function(pred_for_nifti)

    affine, header = fetch_affine_header(subject_name, data_dir)
    pred_nifti = nib.nifti1.Nifti1Image(pred_for_nifti, affine=affine, header=header)
    filename = f'{subject_name}-seg.nii.gz'
    nib.nifti1.save(pred_nifti, os.path.join(save_dir, filename))