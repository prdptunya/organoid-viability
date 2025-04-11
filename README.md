# organoid-viability
# README for Organoid Droplet Viability Pipeline

## Overview
This repository provides a pipeline to:
1. Segment droplets from brightfield and fluorescent microscopy images.
2. Analyze cell viability within each droplet using green fluorescence.
3. Crop valid droplets and save them with viability labels.
4. Train a CNN regression model to predict viability (% dead cells) from images.

The project was originally developed in Google Colab: https://colab.research.google.com/drive/1zzAyokxc-7uNBf-siI-Q60s4YO-Vk9wn?usp=sharing.

---

## Installation
Install dependencies with:
```bash
pip install -r requirements.txt
```

### Required Libraries
- opencv-python
- numpy
- pandas
- torch
- torchvision
- matplotlib
- pillow
- tqdm

---

## 1. Droplet Segmentation and Cropping
**Script:** `segment_and_crop.py`

**Input:** Folder of microscopy images (e.g., `.tif`, `.png`).

**Output:**
- Cropped droplet images (128x128).
- CSV with filenames and viability labels (% dead cells).

**Usage:**
```bash
python segment_and_crop.py --input_folder "/path/to/images" --output_folder "Cropped_Droplets"
```

---

## 2. CNN Model Training
**Script:** `train_viability_cnn.py`

**Input:**
- Folder of cropped droplet images
- CSV file of droplet viability labels

**Output:**
- Trained PyTorch model
- Training/validation loss plot

**Usage:**
```bash
python train_viability_cnn.py --data_dir Cropped_Droplets --label_csv Cropped_Droplets/droplet_viability_labels.csv
```

---

## CNN Architecture
- 2 Convolutional layers (ReLU, MaxPool)
- 1 Fully connected hidden layer
- Output: 1 regression value (log-scaled viability)

Loss function: MSELoss

---

## Citation
If you use this code, please cite the original Colab notebook or credit the repository author.

---

## License
MIT License

---

## Author
Peeradapath Parametphisit

---

For questions or bugs, please open an issue.
