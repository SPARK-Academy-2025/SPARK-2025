# LiMSA-UNet : Lightweight Modality-Selective Attention U-Net for Glioma Segmentation

This repository contains the **official implementation of LiMSA-UNet**, submitted to the **BraTS Lighthouse Challenge 2025**.

LiMSA-UNet is a **lightweight 3D multimodal segmentation model** designed for **glioma sub-region segmentation** from MRI in **resource-constrained clinical settings**, with a focus on **Sub-Saharan African patient populations**. In many low- and middle-income regions, limited access to high-performance computing infrastructure, small and heterogeneous datasets, and under-representation of local patient populations in large public benchmarks hinder the deployment of state-of-the-art deep learning models. This work directly targets these challenges by proposing an efficient, modality-selective architecture that significantly reduces computational and memory requirements while preserving clinically relevant segmentation performance. By explicitly modeling multimodal MRI inputs and leveraging transfer learning from large-scale datasets to African cohorts, LiMSA-UNet aims to bridge the gap between cutting-edge brain tumor segmentation research and practical, deployable solutions for under-resourced clinical environments.

---

## Method Overview

LiMSA-UNet is a compact 3D U-Net–based architecture featuring:

- **Modality-selective encoders** for T1ce, T2, and FLAIR MRI sequences  
- **Attention-based fusion** using 3D CBAM at the bottleneck and skip connections  
- **Residual convolutional blocks** for stable optimization  
- **Low parameter count (~6.4M)** for efficient training and deployment  

The model is **pre-trained on BraTS 2021** and **fine-tuned on BraTS-Africa**, enabling improved generalization to under-represented populations.

---

## Repository Contents

```text
.
├── main.py
├── model.py
├── inference_utils.py
├── model_training_pipeline.ipynb
├── requirements.txt
```

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Data

Intended for use with:

- **BraTS 2021**
- **BraTS-Africa / BraTS-SSA**

Datasets are not included and must be obtained from official BraTS platforms.

---

## License

Released under the **GNU General Public License v3.0 (GPL-3.0)**.

Provided **without warranty**. See:
https://www.gnu.org/licenses/gpl-3.0.html

---

## Citation

If you use this code, please cite:

Sidume F., Moleko N. C., Masalela B. G., Mangwayana P., Kaisara L., Goitsemang R.,  
Rapula T. L., Zhang D., Iorumbur A., Raymond C.  
*LiMSA-UNet: A Lightweight Modality-Selective Attention ResUNet for Brain Tumor Segmentation.*  

---

## Acknowledgements

Developed as part of **SPARK Academy 2025** and submitted to the  
**BraTS Lighthouse Challenge 2025**.

