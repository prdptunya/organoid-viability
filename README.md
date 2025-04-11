# 🧪 Organoid Droplet Viability Estimation

This repository provides code for segmenting microscopy images of droplets, quantifying cell viability (based on green fluorescence), and training a convolutional neural network (CNN) to predict percent dead cells in each droplet.

---

## 📁 Project Structure

```
.
├── Cropped_Droplets/             # Output directory for cropped droplet images and CSV file
├── input/                        # input images directory
├── Organoids_BF.ipynb            # Original Colab notebook
├── assets/                       # Figures and image outputs for documentation
├── README.md                     # This file
```

---

## 🔧 Requirements
Install required packages:
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
This script processes brightfield and fluorescence microscopy images to:
- Detect circular droplets using the Hough Circle Transform.
- Identify cell and green fluorescence regions within droplets.
- Filter droplets with less than 0.1% dead cells.
- Crop 128×128 px images centered around droplets.
- Save the cropped images and their computed viability into a CSV.

**Why this is done:** This preprocessing step is crucial to isolate biologically relevant droplet regions and to generate labeled image data for model training.

**Full Code:**
```python
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

input_folder = "./input_images"
output_crop_dir = "Cropped_Droplets"
CROP_SIZE = 128
os.makedirs(output_crop_dir, exist_ok=True)

all_results = []

def crop_droplet_region(image, x, y, r, size=CROP_SIZE):
    h, w = image.shape[:2]
    left = max(x - r, 0)
    top = max(y - r, 0)
    right = min(x + r, w)
    bottom = min(y + r, h)
    if right <= left or bottom <= top:
        return None
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

for filename in tqdm(os.listdir(input_folder), desc="Processing images"):
    if not filename.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(input_folder, filename)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        continue

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    gray_blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, 1.2, 20,
        param1=50, param2=50, minRadius=50, maxRadius=65
    )

    if circles is None:
        continue

    circles = np.uint16(np.around(circles[0]))

    bf_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bf_mask = cv2.threshold(cv2.GaussianBlur(bf_gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    bf_mask = cv2.morphologyEx(bf_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    for i, (x, y, r) in enumerate(circles, 1):
        droplet_mask = np.zeros_like(bf_mask)
        cv2.circle(droplet_mask, (x, y), r, 255, -1)

        bf_in = cv2.bitwise_and(bf_mask, droplet_mask)
        green_in = cv2.bitwise_and(green_mask, droplet_mask)

        total_area = np.count_nonzero(bf_in)
        dead_area = np.count_nonzero(green_in)

        if total_area == 0 or dead_area == 0:
            continue

        percent_dead = (dead_area / total_area) * 100.0
        if percent_dead < 0.1:
            continue

        crop = crop_droplet_region(img_bgr, x, y, r)
        if crop is not None:
            crop_filename = f"{os.path.splitext(filename)[0]}_droplet_{i:03d}.png"
            cv2.imwrite(os.path.join(output_crop_dir, crop_filename), crop)
            all_results.append({"filename": crop_filename, "Percent_Dead": percent_dead})

pd.DataFrame(all_results).to_csv(os.path.join(output_crop_dir, "droplet_viability_labels.csv"), index=False)
```

---

## 2. CNN Model Training
This script trains a convolutional neural network (CNN) to regress the viability percentage of each droplet.

**Why this is done:** Automating viability assessment accelerates analysis of microfluidic-based organoid experiments and reduces manual variability.

**Full Code:**
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class DropletDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.labels_df.iloc[idx, 0])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        label = np.log1p(self.labels_df.iloc[idx, 1] / 100.0)
        return image, torch.tensor(label, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = DropletDataset(
    csv_file="Cropped_Droplets/droplet_viability_labels.csv",
    root_dir="Cropped_Droplets",
    transform=transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)

model = CNNRegressor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

train_losses, val_losses = [], []
for epoch in range(100):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            val_loss += loss_fn(preds, labels).item()
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss")
plt.grid(True)
plt.show()
```

**CNN Architecture Overview:**
```python
nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 32 * 32, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)
```

**Example Output:**

![Training Droplets](assets/fig3_cropped_droplets.png)

*Fig 3. Cropped droplets used as training input for the CNN model.*

![Loss Curve](assets/fig4_loss_plot.png)

*Fig 4. Training and validation loss over 100 epochs.*

![Prediction Scatter](assets/fig5_pred_vs_actual.png)

*Fig 5. Predicted vs actual % dead cells in the validation set.*

---

## 📊 Output
- Trained model predicts viability from grayscale image input.
- Model evaluation includes reverse log transformation to get back percent dead values.
- Loss curve shows training vs. validation performance over 100 epochs.


