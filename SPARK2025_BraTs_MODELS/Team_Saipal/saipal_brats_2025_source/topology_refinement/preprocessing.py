import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from framework.polynomial import Polynomial, random_mask

size= 2
poly = Polynomial(map_size=size, order=12, dim=3, basis_type='chebyshev')

base_dir = "dataset_2px"
output_dir = "output_2px"
os.makedirs(output_dir, exist_ok=True)

splits = ["Training", "Validation", "Testing"]

def process_split(split):
    # image_dir = os.path.join(base_dir, split, "images")
    mask_dir = os.path.join(base_dir, split, "masks")
    out_split_dir = os.path.join(output_dir, split)
    os.makedirs(out_split_dir, exist_ok=True)

    for filename in os.listdir(mask_dir):
        if not filename.endswith(".nii.gz"):
            continue

        # img_path = os.path.join(mask_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {filename}, skipping.")
            continue

        print(f"Processing: {filename}")

        # img = nib.load(img_path)
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata().astype(np.float32)
        affine = mask.affine

        mask_tensor = torch.tensor(mask_data).long()
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, D, H, W]

        label_masked, masked_tensor, label_skel = random_mask(
            mask_tensor, size, inpaint_type='polynomial', poly=poly, order=2, isCL=True
        )

        output_npy = label_masked.squeeze().cpu().numpy().astype(np.uint8)

        out_nifti = nib.Nifti1Image(output_npy, affine)
        output_filename = os.path.join(out_split_dir, filename.replace("-seg.nii.gz", ".nii.gz"))
        nib.save(out_nifti, output_filename)

        print(f"Saved: {output_filename}")

# Process both Training and Validation
for split in splits:
    process_split(split)

# if __name__ == "__main__":
#     folder_path = "./dataset/Training/masks"
#     rename_seg(folder_path, suffix="-seg")
#     print("Segmentation files renamed successfully.")
#     folder_path = "./dataset/Validation/masks"
#     rename_seg(folder_path, suffix="-seg")
#     print("Segmentation files renamed successfully.")