Cell Segmentation and Counting using Deep U-Net

This project implements a Deep U-Net model to segment and count cell nuclei in microscopy images. The model is trained on the 2018 Data Science Bowl (BBBC038) dataset and includes preprocessing, training, post-processing, and evaluation.

Overview

The pipeline performs:

Automatic dataset download

Image and mask preprocessing

Deep U-Net training

Watershed-based separation of overlapping cells

Counting and performance evaluation

Model

4-level Encoder-Decoder U-Net

Batch Normalization

Dropout in bottleneck

Sigmoid output for binary segmentation

Post-processing uses distance transform + watershed to separate touching cells.

Installation
git clone https://github.com/yourusername/cell-segmentation-unet.git
cd cell-segmentation-unet

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt

Training
python train.py


Output:

best_model.keras

Saved test data (.npy)

Evaluation
python evaluate.py


Generates:

IoU histogram

Bland-Altman plot

Prediction overlays

Requirements

Python 3.8+

TensorFlow 2.x

OpenCV

Scikit-Image

NumPy

Matplotlib
