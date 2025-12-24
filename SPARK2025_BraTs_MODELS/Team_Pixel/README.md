# A Fast, Lightweight nnUNet-based Brain Tumor Segmentation Model Optimized for Low-Resource African Settings (Team South South Nigeria - TeamPixel)
This repository contains the code used to train and run inference with an nnUNet v2–based 3D brain tumor segmentation model using multi-modal MRI. The pipeline was built to simplify the full workflow—from raw data organization to model training, inference, and visualization—while remaining compatible with custom-trained nnUNet checkpoints.

The implementation follows conventions used in BraTS-style datasets and supports four MRI modalities: T1n, T1c, T2w, and T2-FLAIR.
## Overview
The codebase provides:

- Automated dataset preparation for nnUNet v2

- Training and inference wrappers for 3D full-resolution nnUNet

- Support for a custom trainer (MyQuickTrainer)

- Utilities for visualization, localization, and evaluation

- Automatic filename mapping and restoration

This project was developed primarily to support research experimentation and reproducible inference using pretrained or long-running nnUNet models.
## MRI Modalities
The following modality–channel mapping is used throughout the pipeline:

|Channel |	Modality       |
|--------|-----------------|
|0000	   |T1-native (T1n)  |
|0001	   |T1-contrast (T1c)|
|0002	   |T2-weighted (T2w)|
|0003	   |T2-FLAIR (T2f)   |
## Segmentation Labels
|Label	|Description|
|-------|-----------|
|0	    |Background |
|1	    |Whole tumor|
|2	    |Enhancing tumor|
|3	    |Tumor core |

## Dataset Structure
Each subject should be stored in its own folder. File naming must include the modality keyword.
<pre>
  dataset/
├── Patient_001/
│ ├── t1n.nii.gz
│ ├── t1c.nii.gz
│ ├── t2w.nii.gz
│ ├── t2f.nii.gz
│ └── seg.nii.gz
├── Patient_002/
│ └── ...
</pre>
- seg.nii.gz is required only for training

- Inference datasets do not require labels
## Dependencies
- Python ≥ 3.8
- nnUNet v2
- nibabel
- numpy
- OpenCV
- matplotlib
- plotly
Install Python dependencies with:
<pre>
pip install nibabel numpy opencv-python matplotlib plotly
pip install nnunetv2
</pre>
## Training
Training is handled through the nnunet_train function, which automatically:
- Creates the required nnUNet_raw directory structure
- Generates dataset.json
- Copies images and labels into the correct format
- Runs nnUNet preprocessing
- Starts training using the 3D full-resolution configuration
Example usage:
<pre>
  nnunet_train(
    dataset_path="path/to/training_data",
    output_path="path/to/output",
    nnunet_path="path/to/nnunet",
    dataset_name="Dataset001_BrainTumor",
    dataset_id=1
)
</pre>
Checkpoint-based training continuation is supported by passing an existing checkpoint directory.
## Inference
Inference is performed using nnunet_infer. The function:
- Prepares the test dataset
- Runs nnUNet prediction
- Restores original case names
- Saves a mapping file for traceability
Example:
<pre>
  nnunet_infer(
    dataset_path="path/to/test_data",
    output_path="path/to/predictions",
    nnunet_path="path/to/nnunet",
    checkpoint_dir="path/to/checkpoint",
    dataset_name="Dataset001_BrainTumor"
)
</pre>
Final segmentation masks are saved using the original case names.
## Custom Trainer
A lightweight custom trainer (MyQuickTrainer) is included to maintain compatibility with checkpoints trained outside the default nnUNet trainer.

The class inherits directly from nnUNetTrainer and does not modify training logic. It exists solely to allow inference and retraining using previously trained models.
## Visualization and Evaluation
Several utility functions are provided:
- Overlay visualization of segmentation masks on MRI volumes
- Bounding-box localization using OpenCV
 - Slice-wise localized IoU computation
- Interactive 3D volume rendering using Plotly
These tools are intended for qualitative inspection and localized quantitative evaluation.

## LINK TO THE MODEL SOURCE CODE AND WEIGHT  
https://drive.google.com/file/d/18vcBi7H6pdDpcV9OVrANaOd8Dq8wB4Hg/view?usp=sharing

## Notes and Limitations
- The pipeline is configured for 3D full-resolution nnUNet only
- Dataset structure and modality naming are assumed to follow BraTS conventions
- GPU (CUDA) is required for training and recommended for inference
## License
This project is released under the MIT License
## Acknowledgements
- nnUNet v2 (Isensee et al.)
- BraTS Challenge organizers
- MICCAI research community
