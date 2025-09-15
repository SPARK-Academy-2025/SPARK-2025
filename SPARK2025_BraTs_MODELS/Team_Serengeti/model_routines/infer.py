import os
import torch
from ..utils.model_utils import make_dataloader
from ..utils.general_utils import probs_to_preds, save_pred_as_nifti


def infer(data_dir, ckpt_path, out_dir=None, batch_size=1, postprocess_function=None):
    """Uses trained model to make predictions on test data.
    Args:
        data_dir: Directory of test data.
        ckpt_path: Path of trained model.
        out_dir: Directory in which to save predictions. Defaults to None.
        batch_size: Batch size of dataloader. Defaults to 1.
        postprocess_function: The postprocessing function to use. Defaults to None.
    """
    # Set up directories and paths.
    if out_dir is None:
        out_dir = os.getcwd()
    preds_dir = os.path.join(out_dir, 'preds')
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)
        os.system(f'chmod a+rwx {preds_dir}')
    
    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, weights_only=False)
    model = checkpoint['model']
    loss_functions = checkpoint['loss_functions']
    loss_weights = checkpoint['loss_weights']
    training_regions = checkpoint['training_regions']
    epoch = checkpoint['epoch']
    model_sd = checkpoint['model_sd']
    
    # Handle backward compatibility with older checkpoints
    scheduler_patience = checkpoint.get('scheduler_patience', 'Not available (older checkpoint)')
    
    model.load_state_dict(model_sd)
    print(f"Model loaded.")
    
    print("---------------------------------------------------")
    print(f"TRAINING SUMMARY")
    print(f"Model: {model}")
    print(f"Loss functions: {loss_functions}") 
    print(f"Loss weights: {loss_weights}")
    print(f"Training regions: {training_regions}")
    print(f"Epochs trained: {epoch}")
    print(f"Scheduler patience: {scheduler_patience}")
    print("---------------------------------------------------")
    
    print("INFERENCE SUMMARY")
    print(f"Data directory: {data_dir}")
    print(f"Trained model checkpoint path: {ckpt_path}")
    print(f"Out directory: {out_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Postprocess function: {postprocess_function}")
    print("---------------------------------------------------")
    
    test_loader = make_dataloader(data_dir, shuffle=False, mode='test', augment=True, batch_size=batch_size)
    
    print('Inference starts.')
    with torch.no_grad():
        for subject_names, imgs in test_loader:
            model.eval()
            # Move data to GPU.
            imgs = [img.cuda() for img in imgs] # img is B1HWD
            x_in = torch.cat(imgs, dim=1) # x_in is B4HWD
            
            output = model(x_in)
            if isinstance(output, dict):
                output = output['final']
            output = output.float()
            
            preds = probs_to_preds(output, training_regions)
            # preds is B3HWD - each channel is one-hot encoding of a disjoint region
            
            # Iterate over batch and save each prediction.
            for i, subject_name in enumerate(subject_names):
                save_pred_as_nifti(preds[i], preds_dir, data_dir, subject_name, postprocess_function)
    
    print(f'Inference completed. Predictions saved in {preds_dir}.')

if __name__ == '__main__':
    #from ..processing.postprocess import rm_dust_fh
    
    data_dir = '../BraTS2024-SSA-Challenge-TestData'
    #ckpt_path = '/home/mailab/Documents/brats2023_updated/Result/latest_ckpt.pth.tar'
    ckpt_path = '../Result/backup_ckpts/epoch10.pth.tar'
    out_dir = '../prediction'
    postprocess_function = rm_dust_fh
    
    infer(data_dir, ckpt_path, out_dir=out_dir, postprocess_function=postprocess_function)