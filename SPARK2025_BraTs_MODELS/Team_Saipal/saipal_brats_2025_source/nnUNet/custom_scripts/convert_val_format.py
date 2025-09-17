import os, sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nnUNet.nnunetv2.dataset_conversion.val_generator import convert_folder_with_preds_back_to_BraTS_labeling_convention_sequential




validation_preds_folder = "/mnt/c/Machine/Research/Spark 2025/BraTS-2025-all_setups/nnUNetFrame_old/dataset/nnUNet_results/Dataset1137_BraTS2023_SSA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/val_original_train_60"
validation_format_folder = "/mnt/c/Machine/Research/Spark 2025/BraTS-2025-all_setups/nnUNetFrame_old/dataset/nnUNet_results/Dataset1137_BraTS2023_SSA/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/val_original_train_60_format"

convert_folder_with_preds_back_to_BraTS_labeling_convention_sequential(validation_preds_folder, validation_format_folder)