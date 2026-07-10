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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

def prepare_data(data_dir):
    emotion_folders = os.listdir(data_dir)
    images = []
    labels = []
    label_map = {emotion: idx for idx, emotion in enumerate(emotion_folders)}

    for emotion in emotion_folders:
        emotion_path = os.path.join(data_dir, emotion)
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            img = preprocess_image(img_path, augment=True)  # Use augmentation here
            if img is not None:
                images.append(img)
                labels.append(label_map[emotion])

    images = np.array(images).reshape(-1, 48, 48, 1)  # Add channel dimension
    labels = np.array(labels)
    return images, labels

# Prepare data
X, y = prepare_data(train_dir)
y = to_categorical(y, num_classes=len(set(y)))  # One-hot encode labels

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def create_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model

model = create_cnn_model(num_classes=y_train.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=64,
    verbose=1
)

model_path = 'fer_cnn_model.h5'
model.save(model_path)
print(f"Model saved to {model_path}")

# Prepare test data
X_test, y_test = prepare_data(test_dir)
y_test = to_categorical(y_test, num_classes=y_train.shape[1])

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()
