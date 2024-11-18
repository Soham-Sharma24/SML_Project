# SML_Project
# Facial Expression Recognition using CNN
Project Overview
This project focuses on Facial Expression Recognition (FER) using deep learning techniques. It leverages a Convolutional Neural Network (CNN) to classify facial expressions into predefined categories such as happiness, sadness, anger, etc. The model processes images, detects faces, applies preprocessing, and trains on augmented data to achieve accurate recognition. The dataset used is the FER-2013 dataset, a widely used benchmark for emotion recognition tasks.

# Project Features
## Preprocessing Pipeline:

Grayscale conversion, face detection, and resizing.   
Data augmentation: rotation, flipping, zooming, and brightness adjustments.  
Normalization to enhance feature consistency.
## Model Architecture:

A custom CNN architecture with convolutional, pooling, and fully connected layers.  
Regularization techniques like dropout to prevent overfitting.  
## Evaluation Metrics:

Accuracy scores are used for model evaluation.  
Additionaly precision, recall, and F1-score are used for evaluating ML model.  
Visualization of training and validation performance.  

## Feature Extraction Techniques:

Optional use of HOG or LBP for feature-based classification with Random Forest or SVM models.
# Installation and Requirements
## Hardware Requirements
Processor: Multi-core CPU or GPU (NVIDIA preferred).  
RAM: Minimum 8 GB (16 GB recommended for larger datasets).  
GPU: NVIDIA CUDA-enabled GPU with at least 4 GB memory for training.  
## Software Requirements
Python 3.7 or later  
### Required libraries:
TensorFlow/Keras  
NumPy  
OpenCV  
Scikit-learn  
Matplotlib  
Seaborn  
scikit-image  
# Setup Instructions
## Clone the repository:

git clone https://github.com/Soham-Sharma24/SML_Project.git

## Install the required dependencies:

pip install -r requirements.txt

# Usage
Training: Run the provided training script to train the CNN model on FER-2013 or a custom dataset.  
Prediction: Use the trained model to predict facial expressions from images or real-time webcam feeds.  
Feature-Based Models (Optional): Switch to HOG or LBP-based feature extraction for classical machine learning models.  
# Results
Achieved an accuracy of 54% on the validation set.  
Visualized training/validation accuracy and loss curves for performance insights.
