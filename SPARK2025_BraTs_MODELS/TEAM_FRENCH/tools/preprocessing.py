from torch.utils.data import Dataset
import torchio as tio
import os
from glob import glob
import torch.nn.functional as F


NUM_CLASSES = 4  # Update this based on your actual classes

class LazySegmentationDataset_val(Dataset):
    def __init__(self, root_dir: str, transform=None, WT=False):
        self.transform = transform
        
        # Just store file paths
        self.flair_paths = sorted(glob(os.path.join(root_dir, "**", "*t2f*.nii*"), recursive=True))
        self.t1_paths = sorted(glob(os.path.join(root_dir, "**", "*t1n.nii*"), recursive=True))
        self.t1ce_paths = sorted(glob(os.path.join(root_dir, "**", "*t1c*.nii*"), recursive=True))
        self.t2_paths = sorted(glob(os.path.join(root_dir, "**", "*t2w.nii*"), recursive=True))

        assert len(self.flair_paths) == len(self.t1_paths) == len(self.t1ce_paths) == len(self.t2_paths)
            # "Mismatch in number of images"
        
        self.subject_paths = list(zip(self.flair_paths, self.t1_paths, self.t1ce_paths, self.t2_paths))  #, self.mask_paths
    
    def __len__(self):
        return len(self.subject_paths)
    


    def __getitem__(self, idx):
        flair_path, t1_path, t1ce_path, t2_path = self.subject_paths[idx]
    
        subject = tio.Subject(
            flair=tio.ScalarImage(flair_path),
            t1=tio.ScalarImage(t1_path),
            t1ce=tio.ScalarImage(t1ce_path),
            t2=tio.ScalarImage(t2_path),
        )
    
        if self.transform:
            subject = self.transform(subject)
    
       

        return subject


