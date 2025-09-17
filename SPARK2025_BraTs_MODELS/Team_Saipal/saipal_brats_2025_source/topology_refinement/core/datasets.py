import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler

class BraTSDataset(Dataset):
    """
    BraTS-SSA Dataset Class
    """
    def __init__(self,
                 root_dir,
                 mode='train',
                 modalities=["t1c", "t1n", "t2f", "t2w"],
                 target_shape=(128, 128, 128),
                 prediction_dir=None,
                 num_classes=4,
                 return_pretub_mask=False,
                 return_index=False):
        assert mode in ['train', 'test']
        self.root_dir = root_dir
        self.modalities = modalities
        self.target_shape = target_shape
        self.mode = mode
        self.prediction_dir = prediction_dir
        self.num_classes = num_classes
        self.cases = sorted(os.listdir(root_dir))
        self.return_pretub_mask = return_pretub_mask
        self.return_index = return_index

    def __len__(self):
        return len(self.cases)
    
    def getFirstImage(self, idx =  0):
        patient_id = self.cases[idx]
        patient_folder = os.path.join(self.root_dir, patient_id)
        images_folder = os.path.join(patient_folder, "images")

        mod = self.modalities[0]
        filename = f"{patient_id}-{mod}.nii.gz"
        img_path = os.path.join(images_folder, filename)
        return img_path

    def getImage(self, image_filename):
        return os.path.join(self.images_dir, image_filename)

    def getMask(self, mask_filename):
        return os.path.join(self.masks_dir, mask_filename)

    def apply_minmax_scaler(self, image_np, min = -1, max = 1):
        scaler = MinMaxScaler()
        flat = image_np.reshape(min, max)
        scaled_flat = scaler.fit_transform(flat)

        return scaled_flat.reshape(image_np.shape).astype(np.float32)

    def __getitem__(self, idx):
        patient_id = self.cases[idx]

        patient_folder = os.path.join(self.root_dir, patient_id)
        images_folder = os.path.join(patient_folder, "images")
        masks_folder = os.path.join(patient_folder, "masks")

        # Load 4 MRI modalities
        modalities_data = []
        for mod in self.modalities:
            filename = f"{patient_id}-{mod}.nii.gz"
            img_path = os.path.join(images_folder, filename)
            img = nib.load(img_path).get_fdata()
            
            img = self.apply_minmax_scaler(img)
            modalities_data.append(img)

        modalities = torch.tensor(np.stack(modalities_data)).float() # (4, H, W, D)

        # Load pretubated mask (4-class)
        pretub_path = os.path.join(images_folder, f"{patient_id}-seg_perturbed.nii.gz")
        pretub_mask = nib.load(pretub_path).get_fdata()

        pretub_mask = torch.tensor(pretub_mask).long()

        one_hot_mask = F.one_hot(pretub_mask, num_classes=self.num_classes)  # (H, W, D, C)
        one_hot_mask = one_hot_mask.permute(3, 0, 1, 2).float()  # (C, H, W, D)

        input_tensor = torch.cat([modalities, one_hot_mask], dim=0)  # (4 + num_classes, H, W, D)
        
        # Load GT segmentation mask (single channel)
        seg_path = os.path.join(masks_folder, f"{patient_id}-seg.nii.gz")
        seg = nib.load(seg_path).get_fdata()
        seg_tensor = torch.tensor(seg).long()
        
        self.num_classes = self.num_classes or seg_tensor.max().item() + 1
        seg_tensor = F.one_hot(seg_tensor, num_classes=self.num_classes).permute(3, 0, 1, 2).float()

        assert input_tensor.shape[0] == len(self.modalities) + self.num_classes, \
f"Expected {len(self.modalities) + self.num_classes} channels, got {input_tensor.shape[0]}"

        if self.return_index:
            return input_tensor, seg_tensor, idx

        if self.return_pretub_mask:
            return input_tensor, seg_tensor, one_hot_mask
        
        return input_tensor, seg_tensor

class BraTSPredictionDataset(Dataset):
    """
    BraTS-SSA Prediction Dataset Class
    """
    def __init__(self,
                 root_dir,
                 mode='test',
                 modalities=["t1c", "t1n", "t2f", "t2w"],
                 target_shape=(128, 128, 128),
                 prediction_dir=None,
                 num_classes=4,
                 return_pretub_mask=False,
                 return_index=False):
        assert mode in ['train', 'test']
        self.root_dir = root_dir
        self.modalities = modalities
        self.target_shape = target_shape
        self.mode = mode
        self.prediction_dir = prediction_dir
        self.num_classes = num_classes
        self.cases = sorted(os.listdir(root_dir))
        self.return_pretub_mask = return_pretub_mask
        self.return_index = return_index

    def __len__(self):
        return len(self.cases)
    
    def getFirstImage(self, idx =  0):
        patient_id = self.cases[idx]
        patient_folder = os.path.join(self.root_dir, patient_id)
        images_folder = os.path.join(patient_folder, "images")

        mod = self.modalities[0]
        filename = f"{patient_id}-{mod}.nii.gz"
        img_path = os.path.join(images_folder, filename)

        return img_path

    def getImage(self, image_filename):
        return os.path.join(self.images_dir, image_filename)

    def getMask(self, mask_filename):
        return os.path.join(self.masks_dir, mask_filename)

    def apply_minmax_scaler(self, image_np, min = -1, max = 1):
        scaler = MinMaxScaler()
        flat = image_np.reshape(min, max)
        scaled_flat = scaler.fit_transform(flat)

        return scaled_flat.reshape(image_np.shape).astype(np.float32)

    def __getitem__(self, idx):
        patient_id = self.cases[idx]
        patient_folder = os.path.join(self.root_dir, patient_id)
        images_folder = os.path.join(patient_folder, "images")
        masks_folder = os.path.join(patient_folder, "masks")

        # Load modalities directly inside patient_folder
        modalities_data = []
        for mod in self.modalities:
            img_path = os.path.join(images_folder, f"{patient_id}-{mod}.nii.gz")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing modality file: {img_path}")
            img = nib.load(img_path).get_fdata()
            img = self.apply_minmax_scaler(img)
            modalities_data.append(img)
        input_modalities = torch.tensor(np.stack(modalities_data)).float()

        # Load prediction from prediction_dir
        if self.prediction_dir is not None:
            pred_path = os.path.join(self.prediction_dir, f"{patient_id}.nii.gz")
        else:
            # Load pretubated data as prediction for initial testing if predictions_dir is empty
            pred_path = os.path.join(images_folder, f"{patient_id}-seg_perturbed.nii.gz")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Missing prediction file: {pred_path}")
        pred = nib.load(pred_path).get_fdata()
        pred_tensor = torch.tensor(pred).long()
        pred_one_hot = F.one_hot(pred_tensor, num_classes=self.num_classes).permute(3, 0, 1, 2).float()

        input_tensor = torch.cat([input_modalities, pred_one_hot], dim=0)

        # Optional GT segmentation
        seg_path = os.path.join(masks_folder, f"{patient_id}-seg.nii.gz")
        if os.path.exists(seg_path):
            seg = nib.load(seg_path).get_fdata()
            seg_tensor = torch.tensor(seg).long()
            seg_tensor = F.one_hot(seg_tensor, num_classes=self.num_classes).permute(3, 0, 1, 2).float()

            return input_tensor, seg_tensor, patient_id
        else:
            return input_tensor, patient_id
