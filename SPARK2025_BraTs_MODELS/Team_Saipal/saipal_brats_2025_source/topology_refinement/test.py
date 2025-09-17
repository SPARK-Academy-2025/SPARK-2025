import os
import torch
from core.datasets import BraTSPredictionDataset
from core.inference import inference
from core.models import UNet3D
from torch.utils.data import DataLoader
from utils.utils import get_original_image


def run_testing(
    model = None,
    model_input_channels=4,
    model_features=[8, 16, 32],
    test_dir = './data/Testing',
    output_dir = './output',
    prediction_dir = './data/Testing/prediction',
    best_model = None,
    ):
    """
    Main Testing Loop
    """
    original_image = get_original_image(BraTSPredictionDataset(root_dir=test_dir))
    test_dataset = BraTSPredictionDataset(test_dir, "test", prediction_dir=prediction_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model is None:
        model = UNet3D(features=model_features, in_channels=model_input_channels).to(device)
    
    if best_model is None:
        best_model = f'{output_dir}/best_unet3d.pth'
    
    model.load_state_dict(torch.load(best_model))

    inference(model, test_loader, device, original_image=original_image, output_dir=output_dir)
    print("Inference complete.")

if __name__ == "__main__":
    best_model = "output/best_model/best_unet3d_20250804.pth"
    model = None
    model_input_channels=8
    # model_features=[8, 16, 32]
    model_features=[16, 32, 64]
    test_dir = './data/split/ssa/Validation'
    train_dir = './data/split/ssa/Training'
    output_dir = './output/model_output/inference'
    prediction_dir = './data/split/ssa/prediction' # This folder should contain the predicted nii.gz files from ensemble model or any other models to refine the topology of the predictions.

    run_testing(
        model=model,
        model_input_channels=model_input_channels,
        model_features=model_features,
        test_dir=test_dir,
        output_dir=output_dir,
        prediction_dir=prediction_dir,
        best_model=best_model
    )