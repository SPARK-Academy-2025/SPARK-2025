from torch.utils.data import Dataset
import os
import nibabel as nib
import numpy as np
import torch
import random
from scipy.ndimage import map_coordinates, gaussian_filter, zoom
from ..processing.preprocess import znorm_rescale, center_crop

class BratsDataset(Dataset):
    """Dataset class for loading BraTS training and test data with advanced augmentations.
    
    Args:
        data_dir: Directory of training or test data.
        mode: Either 'train' or 'test' specifying which data is being loaded.
        augment: Whether to apply data augmentations (only for training mode).
    """
    
    def __init__(self, data_dir, mode, augment=False, max_subjects=None):
        """Initialize the dataset with augmentation parameters."""
        self.data_dir = data_dir
        self.subject_list = sorted(os.listdir(data_dir))  # sorted for reproducibility
        if max_subjects is not None:
            self.subject_list = self.subject_list[:max_subjects]
        self.mode = mode
        self.augment = augment and mode == 'train'  # Only augment training data
        
        # Enhanced augmentation parameters for stronger augmentation
        self.flip_prob = 0.7          # Increased probability of applying random flips
        self.rotation_range = 20       # Increased degrees (± range for random rotations)
        self.noise_std = 0.15          # Increased standard deviation of Gaussian noise
        self.gamma_range = (0.6, 1.4)  # Wider range for gamma correction
        self.elastic_alpha = (0., 1200.)  # Increased magnitude range for elastic deformation
        self.elastic_sigma = (8., 15.)   # Wider smoothness range for elastic deformation
        self.bias_field_scale = 0.4      # Increased strength of bias field artifact
        
        # New augmentation parameters for stronger augmentation
        self.scale_range = (0.85, 1.15)  # Random scaling factor
        self.contrast_range = (0.7, 1.3)  # Contrast adjustment range
        self.brightness_range = (-0.1, 0.1)  # Brightness adjustment range
        self.blur_prob = 0.2           # Probability of applying Gaussian blur
        self.blur_sigma = (0.5, 1.5)   # Blur sigma range
        self.cutout_prob = 0.15        # Probability of applying cutout
        self.cutout_size = (10, 30)    # Cutout size range
        
        # Modality-specific augmentation parameters
        self.modality_names = ['t1c', 't1n', 't2f', 't2w']
        self.modality_specific_prob = 0.3  # Probability of applying modality-specific augmentations
        
        # T1c-specific (contrast-enhanced): More aggressive contrast/brightness
        self.t1c_contrast_range = (0.6, 1.4)
        self.t1c_brightness_range = (-0.15, 0.15)
        
        # T1n-specific (native): More noise augmentation
        self.t1n_noise_multiplier = 1.5
        
        # T2f-specific (FLAIR): More bias field and gamma correction
        self.t2f_bias_multiplier = 1.3
        self.t2f_gamma_range = (0.5, 1.5)
        
        # T2w-specific: More blur and elastic deformation
        self.t2w_blur_multiplier = 1.5
        self.t2w_elastic_multiplier = 1.2

    def __len__(self):
        return len(self.subject_list)
    
    def load_nifti(self, subject_name, suffix):
        """Loads nifti file for given subject and suffix.
        
        Args:
            subject_name: Name of the subject directory.
            suffix: Modality suffix (e.g., 't1c', 'seg').
            
        Returns:
            Loaded nibabel nifti object.
        """
        nifti_filename = f'{subject_name}-{suffix}.nii.gz'
        nifti_path = os.path.join(self.data_dir, subject_name, nifti_filename)
        return nib.load(nifti_path)
    
    def load_subject_data(self, subject_name):
        """Loads images and segmentation (if in train mode) for a subject.
        
        Args:
            subject_name: Name of the subject directory.
            
        Returns:
            For training: tuple of (modalities_data, seg_data)
            For testing: modalities_data
        """
        modalities_data = []
        for suffix in self.modality_names:  # All 4 standard BraTS modalities
            modality_data = self.load_nifti(subject_name, suffix).get_fdata()
            modalities_data.append(modality_data)

        if self.mode == 'train':
            seg_data = self.load_nifti(subject_name, 'seg').get_fdata()
            return modalities_data, seg_data
        return modalities_data
    
    def apply_augmentations(self, imgs, seg=None):
        """Apply random augmentations to images and segmentation with stronger augmentation.
        
        Args:
            imgs: List of modality images.
            seg: Optional segmentation mask.
            
        Returns:
            Augmented images and segmentation (if provided).
        """
        if not self.augment:
            return imgs, seg
            
        # Random flips (increased probability)
        if random.random() < self.flip_prob:
            axis = random.randint(0, 2)  # Random axis (0, 1, or 2)
            imgs = [np.flip(img, axis=axis) for img in imgs]
            if seg is not None:
                seg = np.flip(seg, axis=axis)
        
        # Random scaling (new augmentation)
        if random.random() < 0.4:
            scale_factor = random.uniform(*self.scale_range)
            imgs, seg = self.scale_images(imgs, seg, scale_factor)
                
        # Random rotation (increased probability)
        if random.random() < 0.4:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            imgs = [self.rotate_image(img, angle) for img in imgs]
            if seg is not None:
                seg = self.rotate_image(seg, angle, is_seg=True)
                
        # Elastic deformations (increased probability)
        if random.random() < 0.4:
            imgs, seg = self.elastic_deform(imgs, seg)
        
        # Cutout augmentation (new)
        if random.random() < self.cutout_prob:
            imgs = self.apply_cutout(imgs)
                
        # Intensity transformations (increased probabilities)
        for i in range(len(imgs)):
            modality = self.modality_names[i]
            
            # Standard intensity augmentations
            if random.random() < 0.4:  # Gaussian noise
                noise_std = self.noise_std
                if modality == 't1n':  # More noise for T1n
                    noise_std *= self.t1n_noise_multiplier
                noise = np.random.normal(0, noise_std, imgs[i].shape)
                imgs[i] = imgs[i] + noise
                
            if random.random() < 0.4:  # Gamma correction
                gamma_range = self.gamma_range
                if modality == 't2f':  # More aggressive gamma for T2f
                    gamma_range = self.t2f_gamma_range
                gamma = random.uniform(*gamma_range)
                imgs[i] = np.sign(imgs[i]) * (np.abs(imgs[i]) ** gamma)
                
            if random.random() < 0.4:  # Bias field artifact
                bias_scale = self.bias_field_scale
                if modality == 't2f':  # Stronger bias field for T2f
                    bias_scale *= self.t2f_bias_multiplier
                imgs[i] = self.add_bias_field(imgs[i], bias_scale)
            
            # New intensity augmentations
            if random.random() < 0.3:  # Contrast adjustment
                contrast_range = self.contrast_range
                if modality == 't1c':  # More aggressive contrast for T1c
                    contrast_range = self.t1c_contrast_range
                contrast = random.uniform(*contrast_range)
                imgs[i] = imgs[i] * contrast
                
            if random.random() < 0.3:  # Brightness adjustment
                brightness_range = self.brightness_range
                if modality == 't1c':  # More aggressive brightness for T1c
                    brightness_range = self.t1c_brightness_range
                brightness = random.uniform(*brightness_range)
                imgs[i] = imgs[i] + brightness
                
            if random.random() < self.blur_prob:  # Gaussian blur
                blur_sigma = random.uniform(*self.blur_sigma)
                if modality == 't2w':  # More blur for T2w
                    blur_sigma *= self.t2w_blur_multiplier
                imgs[i] = gaussian_filter(imgs[i], sigma=blur_sigma)
            
            # Apply modality-specific augmentations
            if random.random() < self.modality_specific_prob:
                imgs[i] = self.apply_modality_specific_augmentation(imgs[i], modality)
                
        return imgs, seg
    
    def apply_modality_specific_augmentation(self, img, modality):
        """Apply modality-specific augmentations.
        
        Args:
            img: Input image.
            modality: Modality name ('t1c', 't1n', 't2f', 't2w').
            
        Returns:
            Augmented image.
        """
        if modality == 't1c':
            # T1c: Simulate contrast agent variations
            if random.random() < 0.5:
                # Simulate uneven contrast enhancement
                enhancement_field = self.create_enhancement_field(img.shape)
                img = img * enhancement_field
                
        elif modality == 't1n':
            # T1n: Add more complex noise patterns
            if random.random() < 0.5:
                # Add Rician noise (common in MRI)
                img = self.add_rician_noise(img)
                
        elif modality == 't2f':
            # T2f: Simulate CSF flow artifacts
            if random.random() < 0.5:
                # Add flow artifacts
                img = self.add_flow_artifacts(img)
                
        elif modality == 't2w':
            # T2w: Add motion artifacts
            if random.random() < 0.5:
                # Simulate motion artifacts
                img = self.add_motion_artifacts(img)
                
        return img
    
    def create_enhancement_field(self, shape):
        """Create a random enhancement field for T1c modality."""
        # Create low-frequency field
        field = np.random.randn(*[s//8 + 1 for s in shape])
        field = zoom(field, 
                    [shape[0]/field.shape[0], 
                     shape[1]/field.shape[1],
                     shape[2]/field.shape[2]], 
                    order=1)
        return 1.0 + 0.3 * np.tanh(field)
    
    def add_rician_noise(self, img):
        """Add Rician noise to image."""
        noise_level = 0.05 * np.std(img)
        noise_real = np.random.normal(0, noise_level, img.shape)
        noise_imag = np.random.normal(0, noise_level, img.shape)
        return np.sqrt((img + noise_real)**2 + noise_imag**2)
    
    def add_flow_artifacts(self, img):
        """Add CSF flow artifacts to FLAIR images."""
        # Create periodic artifacts
        z_coords = np.arange(img.shape[2])
        artifact_pattern = 0.1 * np.sin(2 * np.pi * z_coords / 10)
        artifact_field = np.broadcast_to(artifact_pattern, img.shape)
        return img + img * artifact_field
    
    def add_motion_artifacts(self, img):
        """Add motion artifacts to images."""
        # Simulate motion by applying small random translations
        shift_x = random.uniform(-2, 2)
        shift_y = random.uniform(-2, 2)
        from scipy.ndimage import shift
        return shift(img, [shift_x, shift_y, 0], mode='constant')
    
    def scale_images(self, imgs, seg, scale_factor):
        """Scale images and segmentation by a factor."""
        scaled_imgs = []
        for img in imgs:
            scaled_img = zoom(img, scale_factor, order=3)
            # Crop or pad to original size
            scaled_img = self.resize_to_original(scaled_img, img.shape)
            scaled_imgs.append(scaled_img)
        
        if seg is not None:
            scaled_seg = zoom(seg, scale_factor, order=0)
            seg = self.resize_to_original(scaled_seg, seg.shape)
            
        return scaled_imgs, seg
    
    def resize_to_original(self, img, target_shape):
        """Resize image to target shape by cropping or padding."""
        current_shape = img.shape
        
        # Calculate padding/cropping for each dimension
        result = img.copy()
        for i in range(len(target_shape)):
            diff = target_shape[i] - current_shape[i]
            if diff > 0:  # Need to pad
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width = [(0, 0)] * len(target_shape)
                pad_width[i] = (pad_before, pad_after)
                result = np.pad(result, pad_width, mode='constant')
            elif diff < 0:  # Need to crop
                crop_before = (-diff) // 2
                crop_after = current_shape[i] - (-diff) + crop_before
                slices = [slice(None)] * len(target_shape)
                slices[i] = slice(crop_before, crop_after)
                result = result[tuple(slices)]
        
        return result
    
    def apply_cutout(self, imgs):
        """Apply cutout augmentation to images."""
        cutout_imgs = []
        for img in imgs:
            img_copy = img.copy()
            
            # Random cutout parameters
            cutout_size = random.randint(*self.cutout_size)
            x = random.randint(0, max(1, img.shape[0] - cutout_size))
            y = random.randint(0, max(1, img.shape[1] - cutout_size))
            z = random.randint(0, max(1, img.shape[2] - cutout_size))
            
            # Apply cutout
            img_copy[x:x+cutout_size, y:y+cutout_size, z:z+cutout_size] = 0
            cutout_imgs.append(img_copy)
            
        return cutout_imgs
        
    def elastic_deform(self, imgs, seg):
        """Apply elastic deformation to images and segmentation with stronger deformation.
        
        Args:
            imgs: List of modality images.
            seg: Optional segmentation mask.
            
        Returns:
            Deformed images and segmentation (if provided).
        """
        shape = imgs[0].shape
        alpha = random.uniform(*self.elastic_alpha)  # Deformation magnitude
        sigma = random.uniform(*self.elastic_sigma)  # Deformation smoothness
        
        # Create random displacement fields
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='constant') * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='constant') * alpha
        dz = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='constant') * alpha

        # Create coordinate grid
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        coords = np.array([x + dx, y + dy, z + dz])
        
        # Apply deformation to each modality (cubic interpolation)
        deformed_imgs = [map_coordinates(img, coords, order=3, mode='reflect') for img in imgs]
        
        # Apply to segmentation (nearest neighbor interpolation)
        if seg is not None:
            seg = map_coordinates(seg, coords, order=0, mode='constant')
            
        return deformed_imgs, seg
        
    def add_bias_field(self, image, bias_scale=None):
        """Add MRI bias field artifact to an image.
        
        Args:
            image: Input image to modify.
            bias_scale: Optional bias field scale override.
            
        Returns:
            Image with simulated bias field.
        """
        if bias_scale is None:
            bias_scale = self.bias_field_scale
            
        shape = image.shape
        # Create low-frequency random field
        rand_field = np.random.randn(*[s//16 + 1 for s in shape])
        rand_field = zoom(rand_field, 
                         [shape[0]/rand_field.shape[0], 
                          shape[1]/rand_field.shape[1],
                          shape[2]/rand_field.shape[2]], 
                         order=1)
        
        # Create smooth multiplicative bias field
        bias_field = np.exp(bias_scale * rand_field)
        return image * bias_field
        
    def rotate_image(self, image, angle, is_seg=False):
        """Rotate image by specified angle around axial plane.
        
        Args:
            image: Input image to rotate.
            angle: Rotation angle in degrees.
            is_seg: Whether the image is a segmentation mask.
            
        Returns:
            Rotated image.
        """
        from scipy.ndimage import rotate
        axes = (0, 1)  # Rotate in axial plane
        order = 0 if is_seg else 3  # Nearest neighbor for seg, cubic for images
        return rotate(image, angle, axes=axes, reshape=False, order=order, mode='constant')
    
    def __getitem__(self, idx):
        """Load and process a single subject's data.
        
        Args:
            idx: Index of the subject to load.
            
        Returns:
            For training: (subject_name, modalities, segmentation)
            For testing: (subject_name, modalities)
        """
        subject_name = self.subject_list[idx]

        # Load the data
        if self.mode == 'train':
            imgs, seg = self.load_subject_data(subject_name)
        else:
            imgs = self.load_subject_data(subject_name)
            seg = None

        # Apply augmentations (only for training when augment=True)
        imgs, seg = self.apply_augmentations(imgs, seg)

        # Standard preprocessing
        imgs = [znorm_rescale(img) for img in imgs]  # Normalize each modality
        imgs = [center_crop(img) for img in imgs]    # Center crop
        
        # Convert to tensors
        imgs = [torch.from_numpy(img[None, ...].astype(np.float32)) for img in imgs]  # Add channel dim
        
        if self.mode == 'train':
            seg = center_crop(seg)
            seg = torch.from_numpy(seg[None, ...].astype(np.float32))  # Add channel dim
            return subject_name, imgs, seg
        
        return subject_name, imgs