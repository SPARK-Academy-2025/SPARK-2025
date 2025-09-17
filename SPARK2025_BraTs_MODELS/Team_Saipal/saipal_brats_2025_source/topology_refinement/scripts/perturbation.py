import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import nibabel as nib
import numpy as np
from framework import Polynomial, random_mask
from tqdm import tqdm

def load_brats_mask(path):
    """Load 3D segmentation (classes 0,1,2,3)"""
    img = nib.load(path)
    data = img.get_fdata().astype(np.uint8)
    affine = img.affine
    return data, affine

def process_brats_case_single_class(input_path, output_dir):
    original_mask, affine = load_brats_mask(input_path)
    original_mask = torch.from_numpy(original_mask.astype(np.float32)).unsqueeze(0)  # [1, D, H, W]
    
    poly = Polynomial(map_size=3, order=6, dim=3, basis_type='chebyshev') 
    masked, highlight, _ = random_mask(
        original_mask.unsqueeze(0),
        map_size=3,
        inpaint_type='polynomial_binary',
        poly=poly,
        order=6,
        var=2
    )

    perturbed_mask = masked.squeeze().numpy()
    combined_highlight = highlight.squeeze().numpy()

    case_id = os.path.basename(input_path).split('.nii')[0]
    perturbed_path = os.path.join(output_dir, f"{case_id}_perturbed.nii.gz")
    highlight_path = os.path.join(output_dir, f"{case_id}_highlight.nii.gz")

    nib.save(nib.Nifti1Image(perturbed_mask, affine), perturbed_path)
    nib.save(nib.Nifti1Image(combined_highlight, affine), highlight_path)
    
    return perturbed_path, highlight_path

def process_dataset_folder(input_dir, output_dir):
    """Process all dataset files in a folder"""
    os.makedirs(output_dir, exist_ok=True)
    processed_files = []
    
    # Get all NIfTI files in input directory
    nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    
    for file in tqdm(nii_files, desc="Processing BraTS SSA cases"):
        input_path = os.path.join(input_dir, file)
        perturbed_path, highlight_path = process_brats_case_single_class(input_path, output_dir)
        processed_files.append((perturbed_path, highlight_path))
    
    print(f"\nCompleted processing {len(processed_files)} cases")
    print(f"Perturbed masks saved to: {output_dir}")
    
    return processed_files

if __name__ == "__main__":
    input_directory = "./data/raw/working/masks" 
    output_directory = "./data/raw/ssa/pertubation"
    
    os.makedirs(output_directory, exist_ok=True)
    
    processed = process_dataset_folder(input_directory, output_directory)
    print("\nFirst 5 processed files:")
    for p, h in processed[:5]:
        print(f"- {os.path.basename(p)} (highlight: {os.path.basename(h)})")
