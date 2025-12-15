import os
import nibabel as nib
import numpy as np
import glob
from monai.transforms import LoadImaged, KeepLargestConnectedComponentd, Compose, SaveImaged, FillHolesd
from monai.data import Dataset

def keep_largest_connected_component(source_folder, output_folder):
    transforms = Compose([
        LoadImaged(keys='image', ensure_channel_first=True),
        KeepLargestConnectedComponentd(keys='image',applied_labels=[1,2,3]),
        FillHolesd(keys='image',applied_labels=[1,2,3], connectivity=1),
        SaveImaged(keys='image', output_dir=output_folder, output_postfix='', separate_folder=False)
    ])

    image_paths = sorted(glob.glob(os.path.join(source_folder, "*.nii.gz")))
    data_dicts = []
    for path in image_paths:
        data_dicts.append({
            'image': path
        })
    transformed = Dataset(
        data=data_dicts,
        transform=transforms
    )
    print(f"Found {len(transformed)} images")
    from torch.utils.data import DataLoader
    loader = DataLoader(transformed, batch_size=35, num_workers=4)
    batch = next(iter(loader))



def reorder_masks(source_folder, affine_ref):
    files = os.listdir(source_folder)
    print(f"Found {len(files)} in {source_folder}")
    for file in files:
        filepath = os.path.join(source_folder, file)
        img = nib.load(filepath)
        img = img.get_fdata().transpose(4, 0, 1, 2, 3)
        reshaped = np.zeros((240, 240, 155), dtype=np.uint8)
        reshaped[img[:,:,:,1,0]>0] = 2
        reshaped[img[:,:,:,0,0]>0] = 1
        reshaped[img[:,:,:,2,0]>0] = 3
        affine = nib.load(affine_ref).affine
        nib_image = nib.Nifti1Image(reshaped, affine=affine)
        nib.save(nib_image, filepath)
    print("Work complete!")

if __name__ == "__main__":  
    # Get the source folder from the user
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source_dir', type=str, required=True, help='Path to the source folder containing the images')
    parser.add_argument('--output_dir', type=str, default='./transformed', help='Path to the output folder for transformed images')
    parser.add_argument('--affine_ref', type=str, help='Path to the reference image for affine transformation')
    args = parser.parse_args()

    # Call the function with the provided source folder
    source_folder = args.source_dir
    output_folder = args.output_dir
    affine_ref = args.affine_ref

    if not os.path.exists(affine_ref) or not affine_ref.endswith('.nii.gz'):
        raise ValueError(f"Affine reference file {affine_ref} does not exist or is not a NIfTI file.")

    print(f"Processing images in {source_folder}")
    sample_image = os.path.join(source_folder, os.listdir(source_folder)[0])
    sample_img = nib.load(sample_image)
    print(f"Sample image shape: {sample_img.shape}")
    if len(sample_img.shape) == 5:
        print("Image is 5D, proceeding with processing. Reordering masks first.")
        reorder_masks(source_folder, affine_ref)

    keep_largest_connected_component(source_folder, output_folder)
