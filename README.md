# Cell Segmentation & Counting using Deep U-Net

This project implements a Deep U-Net model for automatic segmentation and counting of cell nuclei in microscopy images. The model is trained on the 2018 Data Science Bowl (BBBC038) dataset and includes preprocessing, augmentation, post-processing, and evaluation.

The goal is to build a complete end-to-end deep learning pipeline for biomedical image segmentation and accurate cell counting.

---

## Project Workflow

The pipeline performs the following steps:

1. Automatically downloads and prepares the BBBC038 dataset  
2. Merges individual mask files into binary segmentation maps  
3. Applies real-time data augmentation (rotation, flipping, zoom)  
4. Trains a Deep U-Net model  
5. Uses watershed post-processing to separate overlapping cells  
6. Evaluates segmentation and counting performance  

---

## Model Architecture

The model is a 5-level Deep U-Net:

- **Encoder:** Conv2D → BatchNorm → ReLU → MaxPooling  
- **Bottleneck:** 512 filters with Dropout (0.3)  
- **Decoder:** Upsampling with skip connections  
- **Output:** Sigmoid activation for pixel-wise prediction  

This architecture preserves spatial resolution while capturing high-level biological features.

---

## Post-Processing (For Accurate Counting)

Segmentation masks often contain touching cells.  
To improve counting accuracy:

- Distance Transform is applied  
- Local maxima are detected  
- Watershed algorithm separates clustered nuclei  
- Connected components are counted  

---

## Installation

```bash
git clone https://github.com/yourusername/cell-segmentation-unet.git
cd cell-segmentation-unet

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt
