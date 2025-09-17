

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch
import numpy as np

class MyQuickTrainer(nnUNetTrainer):
    def __init__(   self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                    device: torch.device = torch.device('cuda')
                ):

        super().__init__(plans, configuration, fold, dataset_json, device)

    