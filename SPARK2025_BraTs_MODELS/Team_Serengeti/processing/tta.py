"""
Test-Time Augmentation (TTA) implementation for improved inference.
"""

import torch
import numpy as np
from torchvision import transforms

class TTATransforms:
    """Class to handle test-time augmentation transforms."""
    
    def __init__(self):
        self.transforms = [
            lambda x: x,  # original
            lambda x: torch.flip(x, [2]),  # flip along depth
            lambda x: torch.flip(x, [3]),  # flip along width
            lambda x: torch.flip(x, [4]),  # flip along height
            lambda x: torch.rot90(x, 1, [3, 4]),  # rotate 90 in plane
            lambda x: torch.rot90(x, 2, [3, 4]),  # rotate 180 in plane
            lambda x: torch.rot90(x, 3, [3, 4]),  # rotate 270 in plane
        ]
        self.inverse_transforms = [
            lambda x: x,  # original
            lambda x: torch.flip(x, [2]),  # flip along depth
            lambda x: torch.flip(x, [3]),  # flip along width
            lambda x: torch.flip(x, [4]),  # flip along height
            lambda x: torch.rot90(x, -1, [3, 4]),  # rotate -90 in plane
            lambda x: torch.rot90(x, -2, [3, 4]),  # rotate -180 in plane
            lambda x: torch.rot90(x, -3, [3, 4]),  # rotate -270 in plane
        ]
    
    def apply_tta(self, model, x_in):
        """Apply test-time augmentation and average predictions."""
        predictions = []
        
        for transform, inv_transform in zip(self.transforms, self.inverse_transforms):
            # Apply augmentation
            augmented_input = transform(x_in)
            
            # Get prediction
            with torch.no_grad():
                output = model(augmented_input)
                output = output.float()
                
            # Reverse augmentation on prediction
            output = inv_transform(output)
            
            predictions.append(output)
        
        # Average predictions
        return torch.mean(torch.stack(predictions), dim=0)