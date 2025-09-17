## End-to-end setup to run BraTS 2025 SSA dataset in nnUNet
```bash
mkdir nnUNetFrame
cd nnUNetFrame

# Create dataset folders
mkdir dataset
cd dataset
mkdir nnUNet_raw_data_base nnUNet_preprocessed RESULTS_FOLDER
cd ..

python -m venv venv
source venv/bin/activate

git clone https://github.com/prabinbohara10/nnUNet.git
cd nnUNet
pip install -e .

# Optional
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git

cd .. # Be inside the nnUNetFrame directory

# env_setup:
# For this env_setup.sh script can be directly run
source MedNeXt/setup_guide/env_setup.sh

# OR run this:
export ORIGINAL_TRAIN_DATA="/home/azureuser/data/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2"
export ORIGINAL_VAL_ONLY_DATA="/home/azureuser/data/BraTS2024-SSA-Challenge-ValidationData" # This is used just to run the prediction/ inference

export nnUNet_raw=... # Absolute Path to nnUNet_raw folder
export nnUNet_preprocessed=... # Absolute Path to nnUNet_preprocessed folder
export nnUNet_results=... # Absolute Path to nnUNet_results folder

export WANDB_PROJECT_NAME=""
export WANDB_RUN_NAME="" # Run name for training experiment in wandb
export WANDB_RUN_NOTES=""

# Create .env file and add WANDB_API_KEY
touch .env # and add WANDB_API_KEY
WANDB_API_KEY = ""

# Steps to run the actual scripts:

# 1. For training: Run python Dataset1137_BraTS23.py
python MedNeXt/nnunet_mednext/dataset_conversion/Dataset1137_BraTS23.py # This formats the data as per the nnUNet and bring all the data to nnUNet_raw folder

# for Brats2025-GLI
python /nnUNet/nnunetv2/dataset_conversion/Dataset1136_BraTS25-GLI.py 

# 2. 
nnUNetv2_plan_and_preprocess -d 1137 --verify_dataset_integrity # This is equivalent to running nnUNetv2_extract_fingerprint, nnUNetv2_plan_experiment and nnUNetv2_preprocess (in that order).
# Puts the output in nnUNet_preprocessed

# 3. 
nnUNetv2_train 1137 3d_fullres 0 -tr nnUNetTrainer_5epochs # different nnUNetTrainer class can be used accordingly.
# examples:
- nnUNetTrainerDC_Focal_HD
- nnUNetTrainerDC_Focal_HD_5epochs


## Steps for prediction:

# 1.
# ORIGINAL_VAL_ONLY_DATA is used for this. Also, change the output folder name in this python script
python nnUNet/nnunetv2/dataset_conversion/Dataset1137_BraTS23_val_only.py

# 2.
nnUNetv2_predict -i /home/azureuser/all_setups/nnUNetFrame/dataset/nnUNet_raw/Dataset1137_BraTS2023_SSA/val_imagesTr_best_model -o /home/azureuser/all_setups/nnUNetFrame/dataset/nnUNet_results/Dataset1137_BraTS2023_SSA/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation_unknown_best_model -d 1137 -c 3d_fullres -f 0

# 3. Converting from nnUNet back to BraTS2023 format
# Change the input and output folder path
python nnUNet/custom_scripts/convert_val_format.py 




```