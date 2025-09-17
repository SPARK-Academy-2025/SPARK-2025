import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.compound_losses import DC_Focal_Hausdorff_Loss
import numpy as np


class nnUNetTrainerDC_Focal_HD(nnUNetTrainer):
    """
    Trainer using combined Dice, Focal, and Hausdorff loss.
    """
    def _build_loss(self):
        # Prepare loss kwargs
        dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'do_bg': self.label_manager.has_regions,
            'smooth': 1e-5,
            'ddp': self.is_ddp
        }
        focal_kwargs = {}
        hausdorff_kwargs = {
            'percentile': 95.0,
            'do_bg': self.label_manager.has_regions,
            'ddp': self.is_ddp
        }
        ignore_label = self.label_manager.ignore_label

        # Instantiate combined loss
        loss = DC_Focal_Hausdorff_Loss(
            dice_kwargs=dice_kwargs,
            focal_kwargs=focal_kwargs,
            hausdorff_kwargs=hausdorff_kwargs,
            weight_dice=1.0,
            weight_focal=1.0,
            weight_hausdorff=1.0,
            ignore_label=ignore_label
        )

        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss


class nnUNetTrainerDC_Focal_HD_5epochs(nnUNetTrainerDC_Focal_HD):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5
