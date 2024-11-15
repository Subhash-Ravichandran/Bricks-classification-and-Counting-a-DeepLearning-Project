# -*- coding: utf-8 -*-
"""Bricks Count.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZYMt7Y8rxjUFipib9cZcbewOu6Z7_piB
"""

# Clone the YOLOv5 repository and navigate to it
!git clone https://github.com/ultralytics/yolov5.git
# %cd yolov5

# Install necessary dependencies
!pip install -r yolov5/requirements.txt
!pip install roboflow

# Unzip the dataset
!unzip -q '/content/Bricks Count.v3i.yolov5pytorch.zip' -d /content/yolov5/dataset/

"""Creating dataset.yaml File"""

# Verify dataset structure
!ls /content/yolov5/dataset/

# Create dataset.yaml configuration
data_yaml = """
path: /content/yolov5/dataset  # Base path to the dataset folder
train: /content/yolov5/dataset/train/images  # Path to training images
val: /content/yolov5/dataset/val/images      # Path to validation images
nc: 2                          # Number of classes
names: ['Stone', 'bricks']      # Class names
"""

with open("/content/yolov5/dataset.yaml", "w") as f:
    f.write(data_yaml)

# Split data into train and validation
import os, shutil, random

base_path = "/content/yolov5/dataset"
train_images = os.path.join(base_path, "train/images")
train_labels = os.path.join(base_path, "train/labels")
val_images = os.path.join(base_path, "val/images")
val_labels = os.path.join(base_path, "val/labels")

os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# Move a subset of images to validation
images = os.listdir(train_images)

val_images_sample = random.sample(images, int(len(images) * 0.2))
for img in val_images_sample:
    shutil.move(os.path.join(train_images, img), os.path.join(val_images, img))
    label = img.replace('.jpg', '.txt')
    shutil.move(os.path.join(train_labels, label), os.path.join(val_labels, label))

# Verify new directory structure
!ls {train_images}
!ls {val_images}

# Train the model
!python yolov5/train.py --img 640 --batch 16 --epochs 300 --data /content/yolov5/dataset.yaml --weights yolov5s.pt --name brick_stone_count --nosave

# Disable WandB logging
os.environ["WANDB_DISABLED"] = "true"

# Model validation
!python yolov5/val.py --weights /content/yolov5/runs/train/brick_stone_count/weights/last.pt --data /content/yolov5/dataset.yaml --img 640

# Run detection on a test image
!python yolov5/detect.py --weights /content/yolov5/runs/train/brick_stone_count/weights/last.pt --img 640 --conf 0.4 --source /content/yolov5/dataset/test/images/rubble_2_jpg.rf.65ce5281bc9a7935cf4c08dabf448802.jpg

!ls /content/yolov5/runs/train/brick_stone_count/weights

!ls /content/yolov5

import torch
# Load and run inference on a new test image
model = torch.hub.load('/content/yolov5', 'custom', path='/content/yolov5/runs/train/brick_stone_count/weights/last.pt', source='local')

img_path = '/content/image (9).jpeg'  # Replace with actual path

results = model(img_path)

# Count bricks and stones
brick_count = sum(1 for obj in results.xyxy[0] if int(obj[-1]) == 1)
stone_count = sum(1 for obj in results.xyxy[0] if int(obj[-1]) == 0)
print(f"Number of Bricks: {brick_count}")
print(f"Number of Stones: {stone_count}")

# Show and save results
results.show()

# Save the model
save_path = '/content/yolov5/runs/train/brick_stone_count/weights/brick_stone_count_model.pt'
torch.save(model, save_path)
print(f"Model saved at {save_path}")