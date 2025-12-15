import numpy as np
import argparse
from light_training.dataloading.dataset import get_kfold_loader
import torch 
import torch.nn as nn 
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.files_helper import save_new_model_and_delete_last
from monai.losses.dice import DiceLoss
set_determinism(123)
import os
import sys
from glob import glob

sys.path.insert(0, os.path.abspath('mamba'))

data_dir = "/home/spark17/TeamLimitless/experiments/segmamba/data/fullres/train"
logdir = f"./logs/segmamba"

model_save_path = os.path.join(logdir, "model")
augmentation = True

env = "pytorch"
max_epoch = 200
batch_size = 2
val_every = 2
num_gpus = 1
device = "cuda:0"
roi_size = [128, 128, 128]
n_splits = 5  # 5-fold cross-validation

def func(m, epochs):
    return np.exp(-10*(1- m / epochs)**2)

class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py", fold=0):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=roi_size,
                                        sw_batch_size=1,
                                        overlap=0.5)
        self.augmentation = augmentation
        from model_segmamba.segmamba import SegMamba

        self.model = SegMamba(in_chans=4,
                        out_chans=4,
                        depths=[2,2,2,2],
                        feat_size=[48, 96, 192, 384])

        self.patch_size = roi_size
        self.best_mean_dice = 0.0
        self.ce = nn.CrossEntropyLoss() 
        self.mse = nn.MSELoss()
        self.train_process = 18
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2, weight_decay=3e-5)
        
        self.scheduler_type = "poly"
        self.cross = nn.CrossEntropyLoss()
        self.fold = fold
        self.load_checkpoint_if_exists(root_path=os.path.join(model_save_path, f"fold_{fold}"), symbol='best')

    def training_step(self, batch):
        image, label = self.get_input(batch)
        
        pred = self.model(image)

        loss = self.cross(pred, label)

        self.log("training_loss", loss, step=self.global_step)

        return loss 
    
    def convert_labels(self, labels):
        ## TC, WT and ET
        result = [(labels == 1) | (labels == 3), (labels == 1) | (labels == 3) | (labels == 2), labels == 3]
        
        return torch.cat(result, dim=1).float()

    
    def get_input(self, batch):
        image = batch["data"]
        label = batch["seg"]
    
        label = label[:, 0].long()
        return image, label

    def cal_metric(self, gt, pred, voxel_spacing=[1.0, 1.0, 1.0]):
        if pred.sum() > 0 and gt.sum() > 0:
            d = dice(pred, gt)
            return np.array([d, 50])
        
        elif gt.sum() == 0 and pred.sum() == 0:
            return np.array([1.0, 50])
        
        else:
            return np.array([0.0, 50])
    
    def validation_step(self, batch):
        image, label = self.get_input(batch)
       
        output = self.model(image)

        output = output.argmax(dim=1)

        output = output[:, None]
        output = self.convert_labels(output)

        label = label[:, None]
        label = self.convert_labels(label)

        output = output.cpu().numpy()
        target = label.cpu().numpy()
        
        dices = []

        c = 3
        for i in range(0, c):
            pred_c = output[:, i]
            target_c = target[:, i]

            cal_dice, _ = self.cal_metric(target_c, pred_c)
            dices.append(cal_dice)
        
        return dices
    
    def validation_end(self, val_outputs):
        dices = val_outputs

        tc, wt, et = dices[0].mean(), dices[1].mean(), dices[2].mean()

        print(f"Fold {self.fold} - dices is {tc, wt, et}")

        mean_dice = (tc + wt + et) / 3 
        
        self.log("tc", tc, step=self.epoch)
        self.log("wt", wt, step=self.epoch)
        self.log("et", et, step=self.epoch)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(model_save_path, f"fold_{self.fold}", 
                                            f"best_model_{mean_dice:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(model_save_path, f"fold_{self.fold}", 
                                        f"final_model_{mean_dice:.4f}.pt"), 
                                        delete_symbol="final_model")

        if (self.epoch + 1) % 100 == 0:
            torch.save(self.model.state_dict(), os.path.join(model_save_path, f"fold_{self.fold}", f"tmp_model_ep{self.epoch}_{mean_dice:.4f}.pt"))

        print(f"Fold {self.fold} - mean_dice is {mean_dice}")
    
    def load_checkpoint_if_exists(self, root_path, symbol='final'):
        if not os.path.isdir(root_path):
            os.makedirs(root_path)
            return
        checkpoint_file = sorted(glob(os.path.join(root_path, f"{symbol}_model_*.pt")))[-1] if glob(os.path.join(root_path, f"{symbol}_model_*.pt")) else None
        if checkpoint_file:
            print(f"Loading checkpoint from {checkpoint_file}")
            self.model.load_state_dict(torch.load(checkpoint_file, weights_only=False))
        else:
            print(f"No checkpoint found with symbol {symbol} in {root_path}. Starting training from scratch.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegMamba model for a specific fold")
    parser.add_argument("--fold", type=int, default=0, help=f"Fold number to train (0 to {n_splits-1})")
    args = parser.parse_args()

    fold = args.fold
    if fold < 0 or fold >= n_splits:
        raise ValueError(f"Fold number must be between 0 and {n_splits-1}, got {fold}")

    print(f"Training is initiated for fold {fold} with data directory: {data_dir}")
    
    # Create fold-specific model save path
    fold_model_save_path = os.path.join(model_save_path, f"fold_{fold}")
    os.makedirs(fold_model_save_path, exist_ok=True)
    
    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir=logdir,
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=17759,  
                           training_script=__file__,
                           fold=fold)
   
    print(f"Fold {fold} - Trainer initialized: {type(trainer)}")

    # Use get_kfold_loader for the specified fold
    train_ds, val_ds, test_ds = get_kfold_loader(data_dir, fold=fold)

    print(f"Fold {fold} - Train dataset size: {len(train_ds)}")
    print(f"Fold {fold} - Validation dataset size: {len(val_ds)}")

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)

    print(f"Fold {fold} - Training completed.")