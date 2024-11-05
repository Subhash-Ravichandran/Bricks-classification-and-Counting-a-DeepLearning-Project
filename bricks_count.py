# -*- coding: utf-8 -*-
"""Bricks Count.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZYMt7Y8rxjUFipib9cZcbewOu6Z7_piB
"""

# Commented out IPython magic to ensure Python compatibility.
# Clone the YOLOv5 repository
!git clone https://github.com/ultralytics/yolov5.git
# %cd yolov5

# Install dependencies
!pip install -r requirements.txt
!pip install roboflow

# # Import and initialize Roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="tcAj36D4QxH3U3iOTqOA")
# project = rf.workspace("subhash-woiz8").project("bricks-count")
# version = project.version(1)
# dataset = version.download("yolov5")



# Unzip dataset
!unzip -q '/content/Bricks Count.v1i.yolov5pytorch.zip' -d /content/yolov5/dataset/

"""Creating dataset.yaml File"""

# Verify dataset structure
!ls /content/yolov5/dataset/

# Create dataset.yaml configuration file
data_yaml = """
path: /content/yolov5/dataset  # Base path to the dataset folder
train: /content/yolov5/dataset/train/images  # Path to training images
val: /content/yolov5/dataset/val/images      # Path to validation images
nc: 1                          # Number of classes
names: ['brick', 'stone']      # Class names
"""

with open("/content/yolov5/dataset.yaml", "w") as f:
    f.write(data_yaml)

# Split dataset into train and validation sets
import os
import shutil
import random

# Define paths
base_path = "/content/yolov5/dataset"
train_images_path = os.path.join(base_path, "train/images")
train_labels_path = os.path.join(base_path, "train/labels")

# Create validation folders
os.makedirs(os.path.join(base_path, "val/images"), exist_ok=True)
os.makedirs(os.path.join(base_path, "val/labels"), exist_ok=True)

# List all training images
images = os.listdir(train_images_path)

# Randomly select a subset of images for validation (20%)
val_size = int(len(images) * 0.2)
val_images = random.sample(images, val_size)

# Move selected images and labels to validation set
for image in val_images:
    shutil.move(os.path.join(train_images_path, image), os.path.join(base_path, "val/images", image))
    label_file = image.replace('.jpg', '.txt')
    shutil.move(os.path.join(train_labels_path, label_file), os.path.join(base_path, "val/labels", label_file))

# Verify the new directory structure
!ls {base_path}/train/images
!ls {base_path}/val/images
!ls {base_path}/train/labels
!ls {base_path}/val/labels

# Training the model
!python train.py --img 640 --batch 16 --epochs 100 --data /content/yolov5/dataset.yaml --weights yolov5s.pt --name brick_stone_count_yolov5 --nosave

# Optional: Custom project name and entity
# !python train.py --img 640 --batch 16 --epochs 100 --data /content/yolov5/dataset.yaml --weights yolov5s.pt --name brick_stone_count_yolov5 --project my_project --entity my_entity

# Disable WandB logging
import os
os.environ["WANDB_DISABLED"] = "true"

!ls runs/train/brick_stone_count_yolov5/weights/

# Model validation
!python val.py --weights runs/train/brick_stone_count_yolov5/weights/last.pt --data /content/yolov5/dataset.yaml --img 640

# Run detection on a test image
!python detect.py --weights runs/train/brick_stone_count_yolov5/weights/last.pt --img 640 --conf 0.1 --source /content/yolov5/dataset/test/images/rubble_2_jpg.rf.811085c1a32ebbd08da66b3014313f92.jpg

!ls /content/yolov5

!ls /content/yolov5

import torch

model = torch.hub.load('/content/yolov5', 'custom', path='/content/yolov5/runs/train/brick_stone_count_yolov5/weights/last.pt', source='local')

# Define path to a new test image
img_path = '/content/burnt-brick-wall-DCDT1K.jpg'  # Replace with actual path

# Run inference
results = model(img_path)

# # Count detected objects (e.g., bricks)
# brick_count = len(results.xyxy[0])  # Counts bounding boxes in the first image
# print(f"Number of bricks detected: {brick_count}")

# Filtering detected objects to count only 'brick' or 'stone'
brick_count = sum(1 for obj in results.xyxy[0] if obj[-1] == 1)  # where 1 corresponds to 'brick'
stone_count = sum(1 for obj in results.xyxy[0] if obj[-1] == 0)  # where 0 corresponds to 'stone'
print(f"Number of bricks detected: {brick_count}")
print(f"Number of stones detected: {stone_count}")

# Show and save results
results.show()

# results.save(save_dir='/content/yolov5/runs/detect/brick_count')