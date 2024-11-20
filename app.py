from flask import Flask, render_template, request, redirect, url_for
from flask_cors import CORS  # Import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Load the model
model_path = 'fer_cnn_model.h5'
model = load_model(model_path)

# Define emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (or configure more granularly if needed)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((48, 48))  # Resize to 48x48
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # Save the uploaded image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image and predict
        preprocessed_image = preprocess_image(filepath)
        prediction = model.predict(preprocessed_image)
        predicted_label = emotion_labels[np.argmax(prediction)]

        # Pass the result to the frontend
        return render_template('result.html', emotion=predicted_label, image_path=filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Ensure the port is 5000 for consistency

