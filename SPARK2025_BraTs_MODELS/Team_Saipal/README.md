# Topology-Driven Fusion of nnU-Net and MedNeXt for Accurate Brain Tumor Segmentation on Sub-Saharan Africa Dataset

This repository contains the official implementation for our BraTS 2025 Challenge submission on the Sub-Saharan Africa Adult Glioma segmentation task.

## Overview

Our approach combines multiple state-of-the-art segmentation architectures with a novel topology refinement module to achieve accurate brain tumor segmentation. The pipeline integrates:

- **nnU-Net**: Self-configuring deep learning framework for medical image segmentation
- **MedNeXt**: Transformer-inspired ConvNet architecture for medical imaging
- **Topology Refinement**: A post-processing module that refines segmentation predictions by preserving topological consistency

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{author_year,
  title={[Paper Title]},
  author={[Author Names]},
  journal={[Journal/Conference Name]},
  year={[Year]},
  doi={[DOI]},
  url={[Paper URL]}
}
```

**Plain text citation:**

[Author Names]. [Paper Title]. [Journal/Conference]. [Year]. [DOI/URL]

## Requirements

- Python >= 3.9 (3.10 recommended)
- PyTorch >= 2.0
- CUDA compatible GPU (recommended: 24GB+ VRAM)

## Installation

Each module has its own setup requirements:

**nnU-Net:**

```bash
cd src/nnUNet
pip install -e .
```

**MedNeXt:**

```bash
cd src/MedNeXt
pip install -e .
```

**Topology Refinement:**

```bash
cd src/topology_refinement
pip install -r requirements.txt
```

For detailed setup instructions, refer to the setup guides in each module:

- [nnU-Net Setup Guide](src/nnUNet/setup_guide/complete_setup.md)
- [MedNeXt Setup Guide](src/MedNeXt/setup_guide/complete_setup.md)

## Dataset

This project uses the BraTS 2025 Sub-Saharan Africa (SSA) Adult Glioma dataset. Each case contains multi-modal MRI scans:

- `*-t1c.nii.gz`: T1-weighted contrast-enhanced
- `*-t1n.nii.gz`: T1-weighted native
- `*-t2f.nii.gz`: T2-weighted FLAIR
- `*-t2w.nii.gz`: T2-weighted
- `*-seg.nii.gz`: Segmentation mask (for training)

### Data Structure

```
data/raw/ssa/
└── BraTS-SSA-XXXXX-XXX/
    ├── BraTS-SSA-XXXXX-XXX-t1c.nii.gz
    ├── BraTS-SSA-XXXXX-XXX-t1n.nii.gz
    ├── BraTS-SSA-XXXXX-XXX-t2f.nii.gz
    ├── BraTS-SSA-XXXXX-XXX-t2w.nii.gz
    └── BraTS-SSA-XXXXX-XXX-seg.nii.gz
```

## Project Structure

```
├── src/
│   ├── nnUNet/                      # nnU-Net segmentation framework
│   │   ├── nnunetv2/                # Core nnU-Net v2 implementation
│   │   ├── setup_guide/             # Installation and setup documentation
│   │   └── README.md
│   │
│   ├── MedNeXt/                     # MedNeXt architecture
│   │   ├── nnunet_mednext/          # MedNeXt integration with nnU-Net
│   │   ├── setup_guide/             # Installation and setup documentation
│   │   └── README.md
│   │
│   └── topology_refinement/         # Topology refinement module
│       ├── core/                    # Core training and model logic
│       │   ├── models.py            # Model architectures
│       │   ├── trainer.py           # Training loop
│       │   └── loss.py              # Loss functions
│       ├── framework/               # Utility framework (clDice, etc.)
│       ├── scripts/                 # Data preparation scripts
│       │   ├── copy_segmentations.py
│       │   ├── perturbation.py
│       │   └── dataset_split.py
│       ├── utils/                   # Helper utilities
│       ├── train.py                 # Training entry point
│       ├── test.py                  # Testing entry point
│       └── README.md
│
├── LICENSE                          # GNU GPL v3 License
└── README.md                        # This file
```

## Usage

### Step 1: Train Base Segmentation Models

Train nnU-Net and MedNeXt models on the BraTS-SSA dataset. Refer to the respective setup guides for detailed instructions:

- [nnU-Net Training](src/nnUNet/README.md)
- [MedNeXt Training](src/MedNeXt/README.md)

### Step 2: Generate Predictions

Generate predictions from both trained models on the validation/test sets.

### Step 3: Topology Refinement

The topology refinement module takes predictions from the base models and refines them to improve topological consistency:

```bash
cd src/topology_refinement

# Prepare dataset
python scripts/copy_segmentations.py
python scripts/perturbation.py
python scripts/dataset_split.py

# Train refinement model
python train.py

# Apply refinement to predictions
python test.py
```

For detailed topology refinement instructions, see [Topology Refinement README](src/topology_refinement/README.md).

## Methods

### Architecture Overview

1. **nnU-Net**: Automatically configures network topology, preprocessing, and training based on dataset properties
2. **MedNeXt**: Employs large kernel convolutions and transformer-inspired design for improved feature extraction
3. **Topology Refinement**: Uses centerline-based loss functions (clDice) to preserve vessel and tumor connectivity

### Segmentation Labels

- **Label 1**: Necrotic Tumor Core (NCR)
- **Label 2**: Peritumoral Edematous/Invaded Tissue (ED)
- **Label 3**: GD-Enhancing Tumor (ET)

## Contact

For questions or collaboration inquiries, please contact: [pralhad.shrestha05@gmail.com](pralhad.shrestha05@gmail.com)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
