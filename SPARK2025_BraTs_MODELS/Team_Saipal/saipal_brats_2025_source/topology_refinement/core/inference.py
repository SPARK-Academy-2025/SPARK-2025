import os
import torch
import nibabel as nib
import numpy as np

def inference(model, loader, device, original_image=None, output_dir='./output'):
    """
    Inference Function
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for _, data in enumerate(loader):
            if len(data) == 3:
                x, y, filename = data
            elif len(data) == 2:
                x, filename = data
                y = None
            else:
                raise ValueError(f"Unexpected number of items returned by dataset: {len(data)}")
            if isinstance(filename, (list, tuple)):
                filename = filename[0]
            
            x = x.to(device)
            out = model(x)

            out = torch.softmax(out, dim=1)  # apply softmax over channels
            pred = torch.argmax(out, dim=1)  # get predicted class index (shape: [B, D, H, W])
            pred_np = pred.cpu().numpy()[0]  # shape: (D, H, W)
            if (y is not None):
                y_np = torch.argmax(y, dim=1)
                y_np = y_np.cpu().numpy()[0]

            nib.save(nib.Nifti1Image(pred_np, original_image.affine, dtype=np.uint8), f'{output_dir}/{filename}.nii.gz')

            print(f"Inference saved: {output_dir}/{filename}")