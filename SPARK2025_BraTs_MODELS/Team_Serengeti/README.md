# MICCAI 2025 BraTS-Africa Challenge: Team Tanzania
## MAPS-Glioma: Modality-Specific Augmentation and Tissue-Adaptive Postprocessing for Robust Glioma Segmentation

## Using this Repository

This repository contains code developed for the BraTS-Africa 2025 Challenge, implementing a deep learning framework that integrates **modality-specific augmentation** and **tissue-adaptive postprocessing** on an optimized 3D U-Net architecture. The framework is specifically designed to address the unique challenges of glioma segmentation in Sub-Saharan African (SSA) populations, where lower-quality MRI scans and distinct tumor characteristics require specialized approaches.

### Key Features

- **Enhanced 3D U-Net architecture** optimized for SSA medical imaging data
- **Modality-specific augmentation** tailored for T1, T1ce, T2, and FLAIR sequences
- **Tissue-adaptive postprocessing** with region-specific refinement
- **Low-resource training strategies** suitable for limited computational infrastructure
- **Comprehensive evaluation metrics** including Dice scores and Hausdorff distances

### BraTS-Africa 2025 Submission

Our final submission achieved the following performance on the validation set:
- **Enhancing Tumor (ET)**: Dice 0.75 ± 0.22, Hausdorff95 11.62 ± 13.68 mm
- **Tumor Core (TC)**: Dice 0.73 ± 0.25, Hausdorff95 13.97 ± 13.12 mm
- **Whole Tumor (WT)**: Dice 0.872 ± 0.17, Hausdorff95 8.86 ± 8.04 mm

Multiple training strategies were explored:
1. Training with only BraTS-Africa SSA data
2. Training with BraTS-Global data and fine-tuning with SSA data
3. Training with combined BraTS-Global and SSA data with modality-specific augmentation
4. Multi-stage training with progressive augmentation strategies

---

## Citation

**Please reference this article if you use this code and its scripts in your research:**

Ayomide B. Oladele, Helena Machibya, Mariam Kaoneka, Frederick Lyimo, Debora Hoza, Immaculata Kafumu, Idris Olalekan, Jeremiah Fadugba, Dong Zhang, Aondona Iorumbu, Raymond Confidence, Nicephorus Rutabasibwa, and Ugumba M. Kwikima. "MAPS-Glioma: Modality-Specific Augmentation and Tissue-Adaptive Postprocessing for Robust Glioma Segmentation in Sub-Saharan Africa." *BraTS-Africa Challenge*, 2025.

```bibtex
@inproceedings{oladele2025maps,
  title={MAPS-Glioma: Modality-Specific Augmentation and Tissue-Adaptive Postprocessing for Robust Glioma Segmentation},
  author={Oladele, Ayomide B and Machibya, Helena and Kaoneka, Mariam and Lyimo, Frederick and Hoza, Debora and Kafumu, Immaculata and Olalekan, Idris and Fadugba, Jeremiah and Zhang, Dong and Iorumbu, Aondona and Confidence, Raymond and Rutabasibwa, Nicephorus and Kwikima, Ugumba M},
  booktitle={BraTS-Africa Challenge},
  year={2025}
}
```

---

## Repository Structure

### Team Tanzania Code

All folders contain original code authored by Team Tanzania for the BraTS-Africa 2025 Challenge. Below are the main components and their purposes.

```
TeamTanzaniaCodes/
├── datasets/
│   ├── __init__.py
│   └── brats_dataset.py              # BraTS-Africa dataset loader and preprocessing pipeline
│
├── losses/
│   ├── __init__.py
│   ├── EdgeLoss3d.py                 # Edge-aware loss for boundary refinement
│   └── general_losses.py             # Dice, Focal, and Combined loss implementations
│
├── model_routines/
│   ├── __init__.py
│   ├── freeze_then_train.py          # Two-stage training with frozen encoder
│   ├── infer.py                      # Inference pipeline with sliding window
│   ├── train.py                      # Standard training routine
│   ├── train_with_val.py             # Training with inline validation
│   └── validate.py                   # Standalone validation script
│
├── models/
│   ├── __init__.py
│   ├── enhanced3dunet.py             # Enhanced 3D U-Net with attention mechanisms
│   └── unet3d.py                     # Base 3D U-Net architecture
│
├── processing/
│   ├── __init__.py
│   ├── plot.py                       # Visualization utilities for segmentation results
│   ├── postprocess.py                # Tissue-adaptive postprocessing pipeline
│   ├── preprocess.py                 # Data preprocessing and normalization
│   └── tta.py                        # Test-time augmentation strategies
│
├── utils/
│   ├── __init__.py
│   ├── general_utils.py              # General helper functions
│   └── model_utils.py                # Model initialization and checkpoint utilities
│
├── code.ipynb                        # Main experimental notebook
├── README.md                         # This file
└── data_README.md                    # Dataset organization instructions
```

---

## Detailed Script Descriptions

### 1. Datasets Module (`datasets/`)

#### `brats_dataset.py`
Custom PyTorch Dataset class for BraTS-Africa 2025 data. Key features:
- Loads multi-modal MRI data (T1, T1ce, T2, FLAIR)
- Handles BraTS-Africa specific file naming conventions (BraTS-AFR-#####)
- Implements efficient caching for preprocessed volumes
- Supports train/validation/test splits with cross-validation folds
- Integrates modality-specific intensity normalization
- Provides on-the-fly data augmentation hooks

**Usage:**
```python
from datasets.brats_dataset import BraTSDataset

dataset = BraTSDataset(
    root_dir="data/processed/train",
    modalities=["t1", "t1ce", "t2", "flair"],
    transform=train_transforms
)
```

---

### 2. Losses Module (`losses/`)

#### `EdgeLoss3d.py`
Implements edge-aware loss function to improve tumor boundary delineation:
- Computes gradients along 3D spatial dimensions
- Emphasizes voxels near tumor boundaries
- Particularly effective for enhancing tumor (ET) segmentation
- Can be combined with Dice loss for balanced training

#### `general_losses.py`
Comprehensive collection of loss functions:
- **DiceLoss**: Standard Dice coefficient loss with smoothing
- **FocalLoss**: Addresses class imbalance in tumor regions
- **CombinedLoss**: Weighted combination of Dice and Focal losses
- **GeneralizedDiceLoss**: Handles multiple classes with class-weighted Dice
- **TverskyLoss**: Adjustable false positive/negative trade-off

**Example:**
```python
from losses.general_losses import CombinedLoss

criterion = CombinedLoss(
    dice_weight=0.5,
    focal_weight=0.5,
    class_weights=[1.0, 2.0, 3.0]
)
```

---

### 3. Model Routines Module (`model_routines/`)

#### `train.py`
Main training script with the following features:
- Configurable hyperparameters via YAML config files
- Automatic mixed precision training (AMP) for efficiency
- Learning rate scheduling (Cosine Annealing, ReduceLROnPlateau)
- Gradient clipping for stability
- Checkpoint saving based on validation metrics
- TensorBoard and Weights & Biases logging

**Usage:**
```bash
python model_routines/train.py --config experiments/config.yaml
```

#### `train_with_val.py`
Enhanced training script with inline validation:
- Validates model every N epochs during training
- Saves best model based on validation Dice score
- Implements early stopping to prevent overfitting
- Generates validation visualizations
- Logs per-class metrics (ET, TC, WT)

#### `freeze_then_train.py`
Two-stage training strategy for transfer learning:
1. **Stage 1**: Train decoder only with frozen encoder
2. **Stage 2**: Fine-tune entire network with lower learning rate

This approach is particularly effective when:
- Pre-training on BraTS-Global data and fine-tuning on SSA data
- Working with limited SSA training samples
- Preventing catastrophic forgetting of learned features

**Usage:**
```bash
python model_routines/freeze_then_train.py \
    --pretrained checkpoints/brats_global.pth \
    --config experiments/finetune_config.yaml
```

#### `validate.py`
Standalone validation script for model evaluation:
- Computes Dice scores for ET, TC, and WT regions
- Calculates Hausdorff95 distances
- Generates confusion matrices and ROC curves
- Produces qualitative visualizations
- Exports metrics to CSV for analysis

**Usage:**
```bash
python model_routines/validate.py \
    --model checkpoints/best_model.pth \
    --data data/processed/val
```

#### `infer.py`
Inference pipeline for test-time prediction:
- Sliding window inference for memory-efficient processing
- Gaussian blending of overlapping patches
- Optional test-time augmentation (TTA)
- Tissue-adaptive postprocessing
- Exports predictions in BraTS submission format

**Usage:**
```bash
python model_routines/infer.py \
    --model checkpoints/best_model.pth \
    --input data/test \
    --output predictions/ \
    --tta  # Optional test-time augmentation
```

---

### 4. Models Module (`models/`)

#### `unet3d.py`
Base 3D U-Net implementation:
- Standard encoder-decoder architecture with skip connections
- Configurable number of feature channels per layer
- Batch normalization and dropout for regularization
- LeakyReLU activations in encoder, ReLU in decoder
- Supports variable input patch sizes

**Architecture:**
```
Input: [B, 4, 128, 128, 128]  # 4 modalities
├── Encoder: [32, 64, 128, 256, 320]
├── Bottleneck: 320 features
├── Decoder: [256, 128, 64, 32]
└── Output: [B, 3, 128, 128, 128]  # 3 classes
```

#### `enhanced3dunet.py`
Enhanced 3D U-Net with advanced components:
- **Attention gates** in skip connections for feature selection
- **Residual blocks** in encoder for deeper networks
- **Deep supervision** with auxiliary outputs
- **Squeeze-and-Excitation (SE) blocks** for channel recalibration
- **Instance normalization** option for improved generalization

This model achieved the best performance on BraTS-Africa validation data.

**Usage:**
```python
from models.enhanced3dunet import Enhanced3DUNet

model = Enhanced3DUNet(
    in_channels=4,
    out_channels=3,
    features=[32, 64, 128, 256, 320],
    use_attention=True,
    deep_supervision=True
)
```

---

### 5. Processing Module (`processing/`)

#### `preprocess.py`
Comprehensive preprocessing pipeline:
- **Skull stripping**: Removes non-brain tissue
- **Resampling**: Ensures consistent voxel spacing
- **Intensity normalization**: Z-score normalization per modality
- **Foreground cropping**: Reduces computational load
- **Co-registration verification**: Checks alignment of modalities
- **Quality control**: Flags corrupted or incomplete scans

**Preprocessing steps:**
1. Load NIfTI files for all modalities
2. Verify image dimensions and orientation
3. Apply skull stripping (optional)
4. Normalize intensities per modality
5. Crop to foreground region
6. Save preprocessed volumes

**Usage:**
```bash
python processing/preprocess.py \
    --input data/raw \
    --output data/processed \
    --modalities t1 t1ce t2 flair
```

#### `postprocess.py`
Tissue-adaptive postprocessing pipeline:
- **Connected component analysis**: Removes small isolated regions
- **Morphological operations**: Closing and opening for smoothing
- **Region-specific thresholding**: Different thresholds for ET, TC, WT
- **Anatomical constraint enforcement**: Ensures ET ⊂ TC ⊂ WT
- **Boundary refinement**: Edge-preserving smoothing

**Key functions:**
```python
from processing.postprocess import apply_postprocessing

# Apply postprocessing to predictions
refined_prediction = apply_postprocessing(
    prediction,
    min_sizes={'ET': 500, 'TC': 1000, 'WT': 2000},
    enforce_hierarchy=True
)
```

#### `plot.py`
Visualization utilities:
- **Multi-slice viewer**: Displays axial, coronal, and sagittal views
- **Overlay visualizations**: Segmentation masks over MRI
- **3D surface rendering**: Tumor visualization in 3D space
- **Metric plots**: Dice scores and loss curves during training
- **Comparison plots**: Ground truth vs. predictions

**Example:**
```python
from processing.plot import visualize_segmentation

visualize_segmentation(
    image=mri_volume,
    ground_truth=gt_mask,
    prediction=pred_mask,
    save_path="results/figures/case_001.png"
)
```

#### `tta.py`
Test-time augmentation (TTA) strategies:
- **Flip augmentations**: X, Y, Z axis flips
- **Rotation augmentations**: 90°, 180°, 270° rotations
- **Ensemble prediction**: Averages predictions from augmented inputs
- **Configurable TTA pipeline**: Select specific augmentations

TTA can improve Dice scores by 1-3% but increases inference time proportionally.

**Usage:**
```python
from processing.tta import TTAWrapper

tta_model = TTAWrapper(
    model,
    transforms=['flip_x', 'flip_y', 'flip_z']
)
prediction = tta_model(input_volume)
```

---

### 6. Utils Module (`utils/`)

#### `general_utils.py`
General-purpose helper functions:
- **Configuration loading**: Parse YAML config files
- **Logging setup**: Configure logging with timestamps
- **File I/O**: Load and save NIfTI files
- **Metrics computation**: Dice, Hausdorff, sensitivity, specificity
- **Data splitting**: Create train/val/test splits with stratification
- **Seed setting**: Ensure reproducibility

#### `model_utils.py`
Model-specific utilities:
- **Model initialization**: Initialize weights (Xavier, Kaiming)
- **Checkpoint management**: Save and load model states
- **Model summary**: Print architecture and parameter counts
- **Optimizer setup**: Configure Adam, AdamW, SGD with momentum
- **Learning rate scheduling**: Cosine annealing, step decay, plateau
- **Mixed precision setup**: Configure automatic mixed precision

---

### 7. Main Notebook (`code.ipynb`)

Interactive Jupyter notebook for:
- **Exploratory data analysis (EDA)**: Visualize BraTS-Africa data distributions
- **Experiment tracking**: Document training runs and results
- **Model comparison**: Compare different architectures and strategies
- **Hyperparameter tuning**: Test different configurations
- **Result visualization**: Generate publication-quality figures
- **Error analysis**: Investigate failure cases

---

## Data Organization

Please refer to `data_README.md` for detailed instructions on:
- Downloading the BraTS-Africa 2025 dataset
- Organizing files according to challenge requirements
- Running preprocessing scripts
- Data quality checks
- Privacy and ethics considerations

### Expected Directory Structure

```
data/
├── raw/
│   ├── BraTS-AFR-00001/
│   │   ├── BraTS-AFR-00001_t1.nii.gz
│   │   ├── BraTS-AFR-00001_t1ce.nii.gz
│   │   ├── BraTS-AFR-00001_t2.nii.gz
│   │   ├── BraTS-AFR-00001_flair.nii.gz
│   │   └── BraTS-AFR-00001_seg.nii.gz
│   └── ...
│
└── processed/
    ├── train/
    ├── val/
    └── test/
```

---

## Workflow: From Data to Submission

### Step 1: Data Preparation
```bash
# Preprocess all raw data
python processing/preprocess.py \
    --input data/raw \
    --output data/processed
```

### Step 2: Training
```bash
# Train with modality-specific augmentation
python model_routines/train_with_val.py \
    --config experiments/config.yaml
```

### Step 3: Validation
```bash
# Evaluate on validation set
python model_routines/validate.py \
    --model checkpoints/best_model.pth \
    --data data/processed/val
```

### Step 4: Inference
```bash
# Generate predictions for test set
python model_routines/infer.py \
    --model checkpoints/best_model.pth \
    --input data/test \
    --output predictions/ \
    --tta
```

### Step 5: Postprocessing
```bash
# Apply tissue-adaptive postprocessing
python processing/postprocess.py \
    --input predictions/ \
    --output predictions_refined/
```

### Step 6: Prepare Submission
Predictions are automatically formatted according to BraTS-Africa submission requirements:
- NIfTI format with `.nii.gz` extension
- Dimensions: 240 × 240 × 155
- Origin at [0, -239, 0]
- Filename format: `BraTS-AFR-{ID}-{timepoint}.nii.gz`

---

## Training Strategies Explored

### Strategy 1: SSA-Only Training
Train exclusively on BraTS-Africa SSA data (60 cases):
- **Pros**: Directly optimized for SSA population characteristics
- **Cons**: Limited training data may lead to overfitting
- **Results**: Competitive on SSA validation but lower generalization

### Strategy 2: Pre-training + Fine-tuning
1. Pre-train on BraTS-Global data (1,251 cases)
2. Fine-tune on BraTS-Africa SSA data
- **Pros**: Leverages large dataset for feature learning
- **Cons**: Risk of catastrophic forgetting
- **Results**: Best balance of generalization and SSA-specific performance

### Strategy 3: Combined Training
Train on mixed BraTS-Global + SSA data with modality-specific augmentation:
- **Pros**: Maximizes data diversity and augmentation strategies
- **Cons**: Requires careful class balancing
- **Results**: Robust performance across different data qualities

### Strategy 4: Progressive Augmentation
Multi-stage training with increasing augmentation intensity:
1. **Stage 1**: Train with standard augmentation
2. **Stage 2**: Add intensity augmentation to mimic SSA data quality
3. **Stage 3**: Fine-tune with heavy augmentation
- **Results**: Improved robustness to image quality variations

---

## Key Design Decisions

### Modality-Specific Augmentation
Unlike standard approaches that apply uniform augmentation across all modalities, MAPS-Glioma implements **modality-specific intensity augmentation**:
- **T1**: Moderate brightness and contrast variations
- **T1ce**: Conservative augmentation to preserve enhancement patterns
- **T2**: Aggressive augmentation to simulate poor contrast
- **FLAIR**: Strong noise augmentation to match real-world SSA scans

This approach better simulates the diverse quality characteristics of SSA medical imaging.

### Tissue-Adaptive Postprocessing
Different tumor regions have distinct characteristics requiring specialized postprocessing:
- **ET**: Small, disconnected regions → aggressive component filtering
- **TC**: Moderate smoothing to preserve core structure
- **WT**: Minimal postprocessing to maintain large region integrity

### Low-Resource Optimization
Several techniques make the model suitable for low-resource settings:
- **Mixed precision training**: Reduces memory by 50%
- **Gradient accumulation**: Simulates larger batch sizes
- **Efficient patch sampling**: Focuses on tumor-containing regions
- **Checkpoint averaging**: Improves robustness without ensemble overhead

---

## Computational Requirements

### Training
- **GPU**: NVIDIA Tesla V100 (16GB) or equivalent
- **RAM**: 32GB minimum
- **Storage**: 100GB for data and checkpoints
- **Time**: ~24 hours for 300 epochs (with mixed precision)

### Inference
- **GPU**: NVIDIA GTX 1080 Ti (11GB) or equivalent
- **RAM**: 16GB minimum
- **Time**: ~30 seconds per case (without TTA)

---

## Results and Performance

### Validation Set Performance

| Tumor Region | Dice Score | Hausdorff95 (mm) | Sensitivity | Specificity |
|--------------|------------|------------------|-------------|-------------|
| ET           | 0.75 ± 0.22 | 11.62 ± 13.68   | 0.78 ± 0.24 | 0.999 ± 0.001 |
| TC           | 0.73 ± 0.25 | 13.97 ± 13.12   | 0.76 ± 0.27 | 0.998 ± 0.002 |
| WT           | 0.872 ± 0.17 | 8.86 ± 8.04    | 0.89 ± 0.15 | 0.997 ± 0.003 |

### Key Findings
1. **WT segmentation** achieved excellent performance (Dice > 0.87)
2. **ET and TC** remain challenging due to ambiguous boundaries
3. **Modality-specific augmentation** improved generalization by 3-5% Dice
4. **Tissue-adaptive postprocessing** reduced false positives by 15-20%

---

## Known Limitations

1. **Small training set**: BraTS-Africa SSA data (60 cases) limits model capacity
2. **Class imbalance**: ET regions are small relative to background
3. **Annotation variability**: Inter-rater differences affect ground truth quality
4. **Computational cost**: TTA increases inference time 8-fold
5. **Generalization**: Performance may vary on external SSA datasets

---

## Future Work

1. **Self-supervised pre-training** on unlabeled SSA MRI data
2. **Domain adaptation** techniques for cross-site generalization
3. **Uncertainty quantification** to flag ambiguous predictions
4. **Lightweight architectures** for mobile/edge deployment
5. **Multi-task learning** with tumor grading and survival prediction

---

## Acknowledgments

We thank the **BraTS-Africa Challenge** organizers for providing the dataset and platform. Special thanks to:
- Teneke Regional Referral Hospital, Tanzania
- Mwananyamala Regional Referral Hospital, Tanzania
- Muhimbili National Hospital, Tanzania
- Muhimbili University of Health and Allied Sciences (MUHAS)
- Medical Artificial Intelligence Laboratory (MAI Lab), Nigeria

This work was supported by clinical collaborators across Tanzania who provided expertise in neuro-oncology and radiology.

---

## License

This project is licensed under the MIT License. See `LICENSE` file for details.

**Note**: The BraTS-Africa dataset is subject to separate terms of use. Users must obtain authorization from the challenge organizers before using the data.

---

## Contact

**Ayomide B. Oladele**  
Medical Artificial Intelligence Laboratory (MAI Lab)  
Email: aoladele@smu.edu  
ORCID: [0009-0002-2883-5459](https://orcid.org/0009-0002-2883-5459)

For questions about the code or methodology, please open an issue on GitHub or contact the authors directly.

---

## Disclosure of Interests

The authors have no competing interests to declare that are relevant to the content of this project.

---

**⭐ If you find this work useful, please cite our paper and star the repository!**
