# SPARK Neuro - BraTS-SSA 2025 Winning Solution

This repository contains the implementation of our winning approach for the MICCAI 2025 Brain Tumor Segmentation Sub-Saharan Africa (BraTS-SSA) Challenge. Our method combines segmentation-aware offline data augmentation with an ensemble of three complementary architectures: MedNeXt, SegMamba, and Residual U-Net, achieving state-of-the-art performance on the BraTS-Africa dataset.

## Key Contributions

- **Segmentation-Aware Data Augmentation**: An offline-data augmentation pipeline which comprises standard geometric and intensity-based transformations together with a custom label-masked elastic deformation, to generate diverse and anatomically plausible data samples.
- **Model Ensemble**: Combination of [MedNeXt](https://github.com/MIC-DKFZ/MedNeXt/tree/main), [SegMamba](https://github.com/ge-xing/SegMamba), and [Residual U-Net](https://github.com/MIC-DKFZ/nnUNet) architectures to leverage complementary strengths.
- **High Performance**: Our best performing model (MedNeXt<sub>1000e</sub>) achieved an average lesion-wise Dice score of 0.86 and Normalized Surface Distance of 0.81, with the ensemble providing a balanced performance across all the tumor subregions.

## Paper

**How We Won BraTS-SSA 2025: Brain Tumor Segmentation in the Sub-Saharan African Population Using Segmentation-Aware Data Augmentation and Model Ensembling**

Claudia Takyi Ankomah, Livingstone Eli Ayivor, Ireneaus Nyame, Leslie Wambo, Patrick Yeboah Bonsu, Aondona Moses Iorumbur, Raymond Confidence, Toufiq Musah.

[arXiv:2510.03568](https://arxiv.org/abs/2510.03568)

Please cite this work if you use our code or methods:

```bibtex
@misc{ankomah2025wonbratsssa2025brain,
      title={How We Won BraTS-SSA 2025: Brain Tumor Segmentation in the Sub-Saharan African Population Using Segmentation-Aware Data Augmentation and Model Ensembling},
      author={Claudia Takyi Ankomah and Livingstone Eli Ayivor and Ireneaus Nyame and Leslie Wambo and Patrick Yeboah Bonsu and Aondona Moses Iorumbur and Raymond Confidence and Toufiq Musah},
      year={2025},
      eprint={2510.03568},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2510.03568},
}
```

## Repository Structure

```
├── brats-transform-augmentations.py    # Segmentation-aware data augmentation script
├── requirements.txt                    # Python dependencies
├── causal-conv1d/                      # Efficient causal convolution implementation
├── mamba/                              # Mamba SSM implementation for SegMamba
├── mednext/                            # MedNeXt architecture (ConvNeXt-based)
├── nnunet_mednext/                     # MedNeXt integration with nnUNet framework
├── nnUNet/                             # nnUNet framework for Residual U-Net
└── README.md                           # This file
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/SPARK-Academy-2025/SPARK-2025.git
cd SPARK-2025/SPARK2025_BraTs_MODELS/SPARK_NeuroAshanti
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

This will install the local packages in editable mode.

## Data Augmentation

The `brats-transform-augmentations.py` script implements segmentation-aware augmentations including:

- Random flips and affine transformations
- Bias field corrections
- Elastic deformations with label masking
- Custom `LabelMaskedElasticDeformation` transform

To generate augmented data:

```python
from brats_transform_augmentations import augment_brats_data

# Augment BraTS-SSA data with 5 new samples per subject
augment_brats_data("path/to/brats_ssa_data", 5)
```

## Training and Inference

Each model can be trained using their respective frameworks:

- **MedNeXt**: Follow the nnUNet pipeline in `nnunet_mednext/`
- **SegMamba**: Use the Mamba SSM modules in `mamba/`
- **Residual U-Net**: Standard nnUNet training in `nnUNet/`

For ensemble inference, combine predictions from all three models using appropriate weighting.

## Using the BraTS Orchestrator

Our winning solution is available through the official BraTS orchestrator for easy inference on BraTS-Africa data. The BraTS orchestrator provides a standardized interface for brain tumor segmentation algorithms.

### Installation

First, install the BraTS package:

```bash
pip install brats
```

### Usage

To use our SPARK Neuro algorithm (BraTS25_1) for segmentation:

```python
from brats import AfricaSegmenter
from brats.constants import AfricaAlgorithms

# Initialize the segmenter
segmenter = AfricaSegmenter(algorithm=AfricaAlgorithms.BraTS25_1, cuda_devices="0")
# Note: these parameters are optional, by default the algorithm will be used on cuda:0

# Perform segmentation on a single case
segmenter.infer_single(
    t1c="path/to/t1c.nii.gz",
    t1n="path/to/t1n.nii.gz",
    t2f="path/to/t2f.nii.gz",
    t2w="path/to/t2w.nii.gz",
    output_file="segmentation.nii.gz",
)
```

For more information about the BraTS orchestrator and available algorithms, visit: https://github.com/BrainLesion/BraTS

## License

This project is licensed under the terms specified in the individual component licenses. Please refer to the LICENSE files in each subdirectory.

## Acknowledgments

<img src="assets\camera_logo.jpg" height="100px" /> <img src="assets\spark_logo.png" height="100px" />

This work was developed as part of the Sprint AI Training for African Medical Imaging Knowledge Translation training program. We are thankful to the organizers and facilitators for the training, mentorship, and resources that supported this work.
