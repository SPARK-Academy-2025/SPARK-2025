
from monai.networks.nets import SwinUNETR

import torch
import torch.nn as nn
import torchio as tio


class SwinUNetRMaskedAutoencoder1(nn.Module):
    def __init__(self, img_size=96, in_channels=4, patch_size=2):
        super().__init__()
        self.encoder = SwinUNETR(
            
            in_channels=in_channels,
            out_channels=4,  # Dummy output
            feature_size=48,
            use_checkpoint=False
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded


def infer_one_subject(model, batch, device):
    images = torch.cat([
        batch['flair'][tio.DATA],
        batch['t1'][tio.DATA],
        batch['t1ce'][tio.DATA],
        batch['t2'][tio.DATA],
        ], dim=1).to(device)

    with torch.no_grad():
        model.eval()
        output = model(images)
        output = torch.argmax(output, dim=1)
    ## Preprocess, Model Inference, Postprocessing Code here...
    return output