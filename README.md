# Bricks Classification & Brick Count

This repository contains two computer vision projects: **Bricks Classification** and **Brick Count**. These projects use deep learning to classify different types of bricks and count the number of bricks in images, automating tasks for industries like brick manufacturing and construction.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Dataset Structure](#dataset-structure)
- [Training Models](#training-models)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Overview

### Bricks Classification
Classifies bricks into different categories based on visual characteristics using a convolutional neural network (CNN), assisting in inventory management and quality control.

### Brick Count
Counts the number of bricks in an image using object detection, aiming to automate counting tasks for logistics and inventory.

## Technologies Used

- Python
- PyTorch
- YOLOv5 (for object detection)
- OpenCV
- Pandas, NumPy
- TensorFlow (optional)
- Jupyter Notebooks (for experiments)

## Project Structure

```plaintext
.
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
├── models/
├── notebooks/
├── results/
└── README.md

Setup and Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/brick-classification-count.git
cd brick-classification-count
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Prepare your dataset as shown in the Dataset Structure section below.

Dataset Structure
Organize your dataset to follow this structure:

plaintext
Copy code
dataset/
├── train/
│   ├── images/       # Training images
│   └── labels/       # Training labels
└── val/
    ├── images/       # Validation images
    └── labels/       # Validation labels
Training Models
To train the models, use the following commands for each project:

Bricks Classification Model
Navigate to the bricks_classification directory and train the model:

python train.py --data dataset.yaml --epochs 50 --batch-size 32
Brick Count Model
Train the brick counting model:

python train.py --img 640 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt --name brick_count_model
Usage
To use the trained models for classification and counting, run the respective commands below.

Classifying Bricks
python classify.py --input path/to/image.jpg --model path/to/classification_model.pt
Counting Bricks

python detect.py --source path/to/image.jpg --weights path/to/count_model.pt --conf 0.5
Results
After training, the evaluation results will be saved in the results directory.

Bricks Classification: Model accuracy and precision for each brick type.
Brick Count: Mean Average Precision (mAP) and counting accuracy.
Acknowledgments
YOLOv5: Used for object detection in brick counting.
PyTorch and TensorFlow: Deep learning frameworks for model development.
OpenCV: For image processing.
