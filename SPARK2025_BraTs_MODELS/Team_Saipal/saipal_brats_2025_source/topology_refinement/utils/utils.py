from datetime import datetime
import nibabel as nib
import numpy as np
from pathlib import Path

def get_timestamps(format='%Y%m%d%H%M%S'):
    """
    Get timestamps using a format
    """
    return datetime.now().strftime(format)

def get_original_image(dataset):
    """
    For inference, use affine from a sample image
    """
    if hasattr(dataset, 'image_files') and dataset.image_files:
        sample_image_path = dataset.getImage(dataset.image_files[0])
        return nib.load(sample_image_path)
    else:
        sample_image_path = dataset.getFirstImage()
        return nib.load(sample_image_path)

def prepare_image_for_logging(img_slice):
    """
    Normalize the image before logging to wandb
    """
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    img_slice = (img_slice * 255).astype(np.uint8)

    # Add channel dimension if missing (e.g., grayscale)
    if img_slice.ndim == 2:
        img_slice = np.expand_dims(img_slice, axis=-1)

    return img_slice

def split_path(path, splitter = '/', return_index = None):
    path_array = path.split(splitter)

    return path_array if (return_index is None) else path_array[return_index]


def get_files(directory, file_index=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a valid directory.")

    files = sorted([f for f in directory.iterdir() if f.is_file()])

    if files:
        return files[file_index] if (file_index is not None) else files
    else:
        print("No files found.")