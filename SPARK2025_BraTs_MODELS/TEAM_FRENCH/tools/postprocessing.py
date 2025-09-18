import os
import torchio as tio
import torch

import nibabel as nib
import numpy as np


def postprocess_segmentation(segmentation_map, batch):
    torch_logits = torch.zeros(1, 240, 240, 155)
    for i in range(segmentation_map.shape[0]):
        label = tio.LabelMap(
            tensor=segmentation_map[i].unsqueeze(0).cpu(),  # FIXED: ensure tensor is on CPU
            affine=batch['flair']['affine'][i],
        )

        one_img_path = batch['flair']['path'][i]
        inverse_crop_pad_transform1 = tio.Pad((2, 2, 2, 2, 3, 3))
        und0_1 = inverse_crop_pad_transform1(label)

        restored_mask = tio.Resample(target=one_img_path)(und0_1)
        torch_logits[i] = torch.tensor(restored_mask[tio.DATA])[0]
        case_ids = ["-".join(batch["flair"]['stem'][0].split("-")[:-1])]
    return torch_logits, case_ids



def save_prediction_like_reference(pred_array, reference_path, save_path):
    """
    Save `pred_array` (H, W, D) as a NIfTI file with the spatial metadata of `reference_path`.
    """
    # Load original image
    ref_nii = nib.load(reference_path)

    # Ensure dtype is correct
    pred_array = pred_array.astype(np.uint8)

    # Create NIfTI with same affine and header
    pred_nii = nib.Nifti1Image(pred_array, affine=ref_nii.affine)

    nib.save(pred_nii, save_path)