import argparse
# from pyexpat import modelections 
from collections import OrderedDict
import os
from tools.read_write import save_nii, load_nii
import torchio as tio
from tools.preprocessing import LazySegmentationDataset_val
from tools.inference import SwinUNetRMaskedAutoencoder1, infer_one_subject
from tools.postprocessing import postprocess_segmentation, save_prediction_like_reference
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "checkpoints/best_swinunetr_warm_restart.pth"  # Path to your model checkpoint
def main(input_dir: str, output_dir: str,model_path=model_path, device=device):
    '''
    Placeholder, fill this in with your inference logic!
    '''
    print(f"Using device: {device}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model path: {model_path}")

    #deifine transoforms
    transforms = tio.Compose([
        tio.ToCanonical(),
        tio.ZNormalization(),
        tio.Resample((2.4, 2.4, 2.2)),
        tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.05, 99.5)),
        tio.CropOrPad((96,96,64)),
    ])
    full_dataset = LazySegmentationDataset_val(root_dir=input_dir, transform=transforms, WT=False)

    #initialize dataloader
    dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=0)



    model= SwinUNetRMaskedAutoencoder1(img_size=96, in_channels=4, patch_size=2).to(device)  # Load your model here
    # Remove 'module.' prefix
    checkpoint = torch.load(model_path , weights_only=False, map_location=device)['model_state_dict']
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)

    print(f"Number of subjects in dataset: {len(full_dataset)}")
    for batch in dataloader:
        predicted_seg = infer_one_subject(model, batch, device)

        mask, case_id = postprocess_segmentation(predicted_seg, batch)
        one_img_path=batch['flair']['path']
        reference_path=one_img_path[0]


        untransformed_image=mask.squeeze()
        
        save_path = os.path.join(output_dir, f"{case_id[0]}.nii.gz")

        save_prediction_like_reference(untransformed_image.numpy(), reference_path, save_path)
        print(f"Saved prediction to {save_path}")



# def infer_one_subject(img_paths: dict):
#     print(f'Running inference using: {img_paths}')
#     ## Load images
#     # images = {contrast:load_nii(pth)[0] for contrast, pth in img_paths.items()}

#     ## Preprocess, Model Inference, Postprocessing Code here...
#     return None


def parse_args():
    parser = argparse.ArgumentParser(description="Run the main processing pipeline.")
    parser.add_argument("-i", "--input", required=True, help="Path to input directory")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory")
    # parser.add_argument("-m", "--model", default=None, help="Path to the model checkpoint file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input
    output_dir = args.output
    # model_path = args.model
    main(input_dir=input_dir, output_dir=output_dir)
