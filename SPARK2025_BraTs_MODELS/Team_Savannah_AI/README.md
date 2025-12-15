# BRAIN-CATS: Calibration-Aware Brain Tumor Segmentation

**SPARK Academy | MICCAI-Aligned Research**

This repository provides the implementation and experimental notebook for BRAIN-CATS (Brain Tumor Reliability-Aware Imaging with Neural Networks using Calibration-Aware Training and Segmentation), developed as part of SPARK Academy by Team Savannah AI. The work addresses brain tumor segmentation challenges in low-resource and Sub-Saharan African (SSA) settings, with a focus on model reliability and calibration, not only accuracy.

The code and experiments in this notebook are referenced in:

> Abba Mohammed, Zulyadaini Muhammad Aminu et al.  
> BRAIN-CATS: Brain Tumor Reliability-Aware Imaging with Neural Networks using Calibration-Aware Training and Segmentation, 2025.

## About BRAIN-CATS

BRAIN-CATS is a calibration-aware brain tumor segmentation framework built on an Attention U-Net architecture, optimized for low-resolution, multi-modal MRI data typical in under-resourced clinical environments.

Unlike conventional segmentation approaches, BRAIN-CATS explicitly integrates confidence calibration into training using the marginal L1 Average Calibration Error (mL1-ACE). This ensures that model predictions are not only accurate but also reliable and trustworthy, a key requirement for clinical deployment.

## Key Features

- Attention U-Net for robust multi-class glioma segmentation
- Calibration-aware training using mL1-ACE loss
- Composite segmentation loss (Dice, BCE, Focal)
- 5-fold cross-validation on BraTS-SSA datasets
- Ensemble inference for stable and consistent predictions
- Designed for limited compute and data availability

## Dataset

The model is trained and evaluated on the ASNR-MICCAI BraTS-SSA datasets, including:

- BraTS 2023 SSA Training Set (60 patients)
- BraTS 2024 SSA Validation Set (35 unseen patients)

**Modalities used:**
- T1
- T1CE
- T2
- FLAIR

**Tumor labels:**
- Background
- Tumor Core (TC)
- Peritumoral Edema (ED)
- Enhancing Tumor (ET)

## Training Strategy

- 2D slice-based training with preprocessing tailored for noisy MRI
- Percentile clipping, z-score normalization, Gaussian denoising
- Data augmentation for robustness to scanner and domain shifts
- Early stopping and Stochastic Weight Averaging (SWA)
- Ensemble inference across folds

## Relation to MICCAI & SPARK Academy

This work was developed within SPARK Academy, an initiative focused on building AI capacity in Africa and advancing medical imaging research aligned with MICCAI standards.

BRAIN-CATS reflects the academy's emphasis on:

- Ethical and reliable AI
- Clinical relevance
- Resource-aware deep learning
- Representation of African datasets in global research

The project aligns with MICCAI Society initiatives such as RISE, mentorship, and capacity building in underrepresented regions.

## License

This project is released as free software under the GNU General Public License v3.0.  
You are free to use, modify, and redistribute the code under the terms of this license.

## Citation

If you use this code or notebook in your research, please cite:
```
Abba Mohammed, Zulyadaini Muhammad Aminu et al.
BRAIN-CATS: Brain Tumor Reliability-Aware Imaging with Neural Networks using
Calibration-Aware Training and Segmentation.
SPARK Academy, 2025.
```

## How to Use the Code

This repository provides a Jupyter Notebook–based implementation for training, validating, and evaluating the BRAIN-CATS model on multi-modal brain MRI data.

### 1. Environment Setup

We recommend using Python 3.9+ with PyTorch and CUDA support.

**Required dependencies:**
```bash
pip install torch torchvision torchaudio
pip install numpy nibabel scipy scikit-learn matplotlib
pip install tqdm einops albumentations
```

**Optional (recommended for GPU training):**
```bash
pip install torchmetrics
```

The code has been tested on Linux-based GPU environments (e.g., RunPod, Colab, local CUDA setups).

### 2. Dataset Preparation

Organize the dataset in the following structure:
```
dataset/
├── imagesTr/
│   ├── BraTS-SSA_001_t1.nii.gz
│   ├── BraTS-SSA_001_t1ce.nii.gz
│   ├── BraTS-SSA_001_t2.nii.gz
│   └── BraTS-SSA_001_flair.nii.gz
├── labelsTr/
│   └── BraTS-SSA_001_seg.nii.gz
```

Each case must contain the four MRI modalities:

- T1
- T1CE
- T2
- FLAIR

Segmentation labels follow the BraTS convention.

### 3. Running the Notebook

Open the main notebook:
```
team-savannah_TeamLeader_Code_Cross-Entropy_Loss_with_ML1_ACE.ipynb
```

The notebook is organized into clearly labeled sections:

- Imports & Configuration
- Dataset Loading & Preprocessing
- Model Architecture (Attention U-Net)
- Loss Functions & Calibration Metrics
- Training Loop
- Validation & Evaluation
- Inference & Visualization

Run the cells top-to-bottom to reproduce results.

## Implementation Details

### Model Architecture

BRAIN-CATS uses an Attention U-Net architecture designed to:

- Focus on tumor-relevant regions
- Suppress background noise
- Handle heterogeneity in SSA MRI scans

Attention gates are applied at skip connections to improve feature fusion.

### Loss Functions

The training objective combines segmentation accuracy and calibration awareness.

**Segmentation Loss:**
- Cross-Entropy Loss (multi-class)
- Dice Loss (class overlap optimization)

**Calibration Metric:**
- Marginal L1 Average Calibration Error (mL1-ACE)

mL1-ACE is used to quantify prediction reliability by measuring the gap between confidence and empirical accuracy across bins.

Example implementation:
```python
total_loss = segmentation_loss
calibration_error = ml1_ace(predictions, targets)
```

While mL1-ACE is not backpropagated, it is monitored to ensure trustworthy predictions.

### Training Strategy

- 2D slice-wise training
- Z-score normalization per modality
- Percentile intensity clipping
- On-the-fly data augmentation
- Early stopping based on validation Dice
- Stochastic Weight Averaging (SWA) for improved generalization

### Evaluation Metrics

The following metrics are reported:

- Dice Similarity Coefficient (DSC)
- Precision and Recall
- Per-class accuracy
- mL1-ACE calibration score

Results are computed per tumor sub-region:

- Enhancing Tumor (ET)
- Tumor Core (TC)
- Whole Tumor (WT)

### Inference

Inference is performed using:

- Sliding-window prediction
- Ensemble averaging across cross-validation folds

Predicted segmentation maps are visualized alongside ground truth for qualitative assessment.

## Notes on Reproducibility

- Random seeds are fixed where applicable
- Cross-validation splits are consistent across experiments
- All preprocessing steps are explicitly defined in the notebook

## Intended Use

This code is intended for:

- Research and educational purposes
- Benchmarking calibration-aware segmentation methods
- Exploring AI reliability in low-resource medical imaging settings

It is **not** intended for direct clinical deployment without further validation.

## Acknowledgements

We gratefully acknowledge SPARK Academy, CAMERA, and clinical collaborators for their support, as well as MICCAI-aligned mentors and facilitators who contributed to the technical and scientific development of this work.