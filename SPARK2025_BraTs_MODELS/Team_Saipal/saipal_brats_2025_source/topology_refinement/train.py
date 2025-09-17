import os
import shutil
import torch
import torch.optim as optim
import wandb
from core.datasets import BraTSDataset
from core.loss import getCriterion
from core.models import UNet3D
from core.trainer import eval_epoch, train_epoch
from args import parse_args
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils.model_utils import count_parameters
from utils.trainer_utils import load_checkpoint, save_checkpoint
from utils.utils import get_timestamps
from utils.wandb_utils import init_wandb, log_test_image, log_train_image
from test import run_testing

def run_training(
    train_dir='./data/Training',
    val_dir='./data/Validation',
    test_dir=None,
    prediction_dir = None,
    output_dir='./output',
    model_input_channels=4,
    model_features=[8, 16, 32],
    epochs=500,
    batch_size=1,
    lr=1e-4,
    wandb_project="unet3d-brats2025",
    wandb_run_name="run_unet3d",
    wandb_init=True,
    args=None
):
    """
    Main Training Loop
    """
    run_name = f"{wandb_run_name}_{epochs}epochs_{get_timestamps('%Y%m%d_%H%M%S')}"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if wandb_init:
        init_wandb(
            project=wandb_project,
            run_name=run_name,
            config={"lr": lr, "epochs": epochs, "batch_size": batch_size, "features": model_features},
            resume=args.resume if args else False,
            id_file=f"{checkpoint_dir}/wandb_run_id.txt"
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = BraTSDataset(root_dir=train_dir)
    val_dataset = BraTSDataset(val_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # For logging same training image in the log
    logging_train_dataset = BraTSDataset(train_dir, return_index=True)
    train_sample = next(iter(DataLoader(logging_train_dataset, batch_size=1, shuffle=False)))
    sample_train_input, sample_train_target, train_sample_no = train_sample[0].to(device), train_sample[1].to(device), train_sample[2].item()
    
    # For logging same testing image in the log
    logging_test_dataset = BraTSDataset(val_dir, return_index=True)
    test_sample = next(iter(DataLoader(logging_test_dataset, batch_size=1, shuffle=False)))
    sample_test_input, sample_test_target, test_sample_no = test_sample[0].to(device), test_sample[1].to(device), test_sample[2].item()
    
    model = UNet3D(features=model_features, in_channels=model_input_channels).to(device)
    # print(count_parameters(model))
    criterion = getCriterion()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=25)
    best_val_loss = float('inf')

    checkpoint_path = f"{checkpoint_dir}/checkpoint.pth"
    best_model_path = f'{run_dir}/best_unet3d.pth'
    output_best_model_dir = f'{output_dir}/best_model/'
    os.makedirs(output_best_model_dir, exist_ok=True)
    
    start_epoch = 0

    if (os.path.exists(checkpoint_path) and args is not None and args.resume is True):
        print("Loading checkpoint...")
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming from epoch {start_epoch}, loss {best_val_loss}")

    for epoch in range(start_epoch, epochs):
        train_loss, train_loss_logging = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_loss_logging = eval_epoch(model, val_loader, criterion, device, epoch)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        train_log = log_train_image(model, sample_train_input, sample_train_target, train_sample_no, 117)
        val_log = log_test_image(model, sample_test_input, sample_test_target, test_sample_no, 78)
        if wandb.run is not None:
            wandb.log({
                **train_log,
                **val_log,
                **train_loss_logging,
                **val_loss_logging,
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
                "total/epoch": epoch
            }, step=epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved.")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path)

    print("Training complete.")
    shutil.copy2(best_model_path, output_best_model_dir)
    print(f"Best model saved to {output_best_model_dir}.")

    # Inference
    if (test_dir is not None):
        run_testing(
            model,
            model_input_channels,
            model_features,
            test_dir=test_dir,
            output_dir=run_dir,
            prediction_dir=prediction_dir
        )
    if wandb_init:
        wandb.finish()



# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    args = parse_args()
    if os.path.exists("./wandb.json"):
        import json
        with open('./wandb.json', 'r') as file:
            data = json.load(file)
        os.environ["WANDB_API_KEY"] = data["work"]

    run_training(
        epochs=500,
        batch_size=4,
        wandb_project="topology-unet3d",
        wandb_init=False, # Change to True if you use wandb
        lr=1e-3,
        model_input_channels=8,
        # model_features=[8, 16, 32],
        model_features=[16, 32, 64],
        wandb_run_name="run_train_unet3d",
        train_dir='./data/split/ssa/Training',
        val_dir='./data/split/ssa/Validation',
        test_dir="./data/split/ssa/Testing",
        # prediction_dir = './data/split/ssa/prediction', # Use the prediction directory if you have existing predictions to use in testing
        prediction_dir = None,
        args=args
)
