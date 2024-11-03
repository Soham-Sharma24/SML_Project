import zipfile
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

zip_file_path = 'fer2013.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('fer2013_data')

train_dir = 'fer2013_data/train'
test_dir = 'fer2013_data/test'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def augment_image(img):
    rows, cols = img.shape
    
    # Random rotation
    angle = np.random.uniform(-15, 15)
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_matrix, (cols, rows))

    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)

    # Random zoom
    zoom_scale = np.random.uniform(0.9, 1.1)
    zoom_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, zoom_scale)
    img = cv2.warpAffine(img, zoom_matrix, (cols, rows))

    # Random brightness adjustment
    brightness_adjustment = np.random.uniform(0.8, 1.2)
    img = np.clip(img * brightness_adjustment, 0, 255).astype(np.uint8)
    
    return img

# Function to detect faces, resize, normalize, augment, and return the processed image
def preprocess_image(image_path, target_size=(48, 48), augment=False):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None  # Skip if image is not found
    # Detect faces
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # If no faces are detected, proceed with the entire image
    if len(faces) == 0:
        cropped_img = img  # Use the whole image if no face is detected
    else:
        # Crop the first detected face region
        (x, y, w, h) = faces[0]
        cropped_img = img[y:y+h, x:x+w]
    # Resize to target dimensions
    resized_img = cv2.resize(cropped_img, target_size)
    # Apply data augmentation if specified
    if augment:
        resized_img = augment_image(resized_img)
    # Normalize the pixel values to the range [0, 1]
    normalized_img = resized_img / 255.0
    return normalized_img

def visualize_processed_images(data_dir, sample_size=5, augment=False):
    emotion_folders = os.listdir(data_dir)
    fig, ax = plt.subplots(len(emotion_folders), sample_size, figsize=(20, 3 * len(emotion_folders)))

    for row, emotion in enumerate(emotion_folders):
        emotion_path = os.path.join(data_dir, emotion)
        images = os.listdir(emotion_path)[:sample_size]
        
        for col, img_name in enumerate(images):
            img_path = os.path.join(emotion_path, img_name)
            processed_img = preprocess_image(img_path, augment=augment)
            
            if processed_img is not None:
                ax[row, col].imshow(processed_img, cmap='gray')
                ax[row, col].set_title(emotion)
                ax[row, col].axis('off')
            else:
                ax[row, col].axis('off')  # In case of missing image

    plt.tight_layout()
    plt.show()

print("Train Set with Augmentation:")
visualize_processed_images(train_dir, sample_size=5, augment=True)

print("Test Set without Augmentation:")
visualize_processed_images(test_dir, sample_size=5, augment=False)

def plot_class_distribution(data_dir):
    emotion_folders = os.listdir(data_dir)
    class_counts = {}

    # Count the number of images in each emotion category
    for emotion in emotion_folders:
        emotion_path = os.path.join(data_dir, emotion)
        if os.path.isdir(emotion_path):
            class_counts[emotion] = len(os.listdir(emotion_path))

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel('Emotion Category')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution of Images in Each Emotion Category')
    plt.xticks(rotation=45)
    plt.show()

print("Train Set Class Distribution:")
plot_class_distribution(train_dir)

print("Test Set Class Distribution:")
plot_class_distribution(test_dir)