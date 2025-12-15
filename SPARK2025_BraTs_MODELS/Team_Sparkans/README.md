# MICCAI 2025 Brain Tumour Segmentation Challenge: Team SPARKANS

This repository contains the full inference and ensembling pipeline used by **Team SPARKANS** for the **MICCAI 2025 Brain Tumour Segmentation Challenge**.

Our final submission is based on an **ensemble of nnU-Net and MedNeXt models**, with custom architectural and inference modifications.

---

## Repository Overview

This repository contains newly developed code as well as slightly modified version of the Optimised U-Net model as found in the Nvidia Deep Learning Examples repository for nnUnet v2 and MedNeXt.

BraTS2025 final submission only used the Optimised framework. A variety of models were trained and evaluated including training with only SSA data, training with only BraTS-GLIOMA data and training with only BraTS-GLIOMA data and then fine-tuning the model with SSA data.

The newly developed code was created in order to further explore the effect of additional data augmentations and class weighted ensembling on the generalisability of this framework. 
```
├── networks/                 # Custom-modified nnU-Net and MedNeXt network definitions
├── models/                   # Trained model weights and checkpoints
├── prepare_*.py               # Preprocessing and inference preparation scripts
├── convert_mednext_npz.py     # Converts MedNeXt outputs for ensembling
```

---

## Model Architecture

### Ensemble Strategy

The final prediction is obtained by **ensembling outputs from two independent segmentation frameworks**:

* **nnU-Net v2** (custom-modified)
* **MedNeXt** (custom-modified)

Both models are trained independently and produce voxel-wise multi-class probability maps. These probability maps are combined during post-processing to generate the final segmentation.

### Custom Network Modifications

All changes are located in the `networks/` directory. These include:

Custom augmentation strategies.

Class weighted ensembling strategy.

---

## Trained Models

All trained checkpoints are stored in the `models/` directory.

---

## Inference Pipeline

### Preparation Scripts (`prepare_*.py`)

The `prepare` scripts are responsible for **preparing data and configuration files prior to inference**. This typically includes:

* Dataset structure validation for inference.

### MedNeXt Conversion (`convert_mednext_npz.py`)

MedNeXt produces predictions in its native format after inference. The script:

```
convert_mednext_npz.py
```

* Loads MedNeXt prediction outputs
* Converts them into nnU-Net–compatible `.npz` probability maps
* Aligns class ordering and spatial metadata

This conversion step is **required before ensembling** MedNeXt outputs with nnU-Net predictions.

---

## Ensembling

Ensembling is performed by:

1. Loading nnU-Net probability maps
2. Loading converted MedNeXt probability maps
3. Averaging (or weighted averaging) class-wise probabilities
4. Applying argmax and post-processing

The final output is a BraTS-compliant segmentation mask.


---

## Notes

* This repository is **research code** provided for transparency and reproducibility.
* Paths, environment variables, and dataset locations may need to be adapted to your system.
* For challenge submission, ensure output files follow the MICCAI BraTS naming and formatting requirements.
* "sparkans.sh" file contains commands run for final test submission of BraTS2025 Challange.

---

## Team SPARKANS

MICCAI 2025 Brain Tumour Segmentation Challenge
