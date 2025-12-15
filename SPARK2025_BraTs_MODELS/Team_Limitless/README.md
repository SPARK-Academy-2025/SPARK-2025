# Team Rwanda – Limitless (BraTS/SPARK 2025)

This repository aggregates three open source projects we used for our contribution to the BRATS 2025 challenge: NN-Unet, MedNext and SegMamba. Our idea was to train each of the models independently and ensemble their output using the STAPLE ensemble.

Each of the three model types can be used independently by following the README.md and documentation for each of them. Once training has finished, you can copy results of the model you are training into `/tmp` directory or modify the [ensemble.py](./ensemble.py) to update the paths to your model outputs. Files `mednexttrain`, `mednextval`, `nnunettrain` and `nnunetval` implements classes to run help run the respective models on compute canada clusters. In addition to the original `SegMamba/3_train.py`, we added `SegMamba/33_train.py` in which we implement five fold cross validation. The directory structure is described in details below.

## Repository structure

```
.
├─ README.md
|─ ensemble.py
|─ mednexttrain.py
|─ mednextval.py
|─ nnunettrain.py
|─ nnunetval.py
├─ nnUNet/
├─ SegMamba/
└─ MedNeXt/
```

### nnUNet/
Contains source code for nnUnetV2. See `nnUnet/README.md` and `nnUnet/documentation/` for general MedNext usage

### SegMamba/
SegMamba by default divides the dataset into a 70/10/20 train-validation-test split. Due to a dataset of only 60 cases, this feature was unnecessary for us. In `SegMamba/3_train.py`, we impliment training with 5-fold cross validation uaing 80% of data for training and 20% for validation for every fold. SegMamba also have a tendency to create noise. We used `SegMamba/9_postprocess.py` to polish the output by keeping only the largest connected component.

See `SegMamba/README.md` for more details on SegMamba.

### MedNeXt/
We only made changes in `nnunet_mednext/dataset_conversion/Task137_BraTS_2021.py` to make the dataset converter use the correct dataset file naming structure for fingerprinting and preprocessing.

Otherwise, see `MedNeXt/README.md` and `MedNeXt/documentation/` for general MedNext usage.

## How to use
- Each subfolder is a self-contained project. Install and run inside the respective directory.
- Typical setup:
  - nnUNet:
    - `cd nnUNet && pip install -e .`
    - Follow `documentation/setting_up_paths.md` and `installation_instructions.md`
  - SegMamba:
    - `cd SegMamba && pip install -r requirements.txt`
    - Use the provided scripts (`3_train.py`, `4_predict.py`) as per `SegMamba/README.md`
  - MedNeXt:
    - `cd MedNeXt && pip install -e .`
    - Follow documentation for paths, dataset conversion, training, and inference

## Pretrained Weights
For SegMamba, we provide checkpoint for the best of the five folds we trained and can be found [here](https://drive.google.com/file/d/1yIiZXMN9UECilmvfCIKkv-57-YBxD-B2/). We also provide weights for nnUnet trained on all 60 cases of the BRATS SSA dataset [here](https://drive.google.com/file/d/1WV-te4r1JcNk1DenrU4XpATBKGVvmD8T/). 