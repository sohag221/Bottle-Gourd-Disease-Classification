from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Load your stacking model and label encoder
model = joblib.load("stacking_ensemble_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Simulate loading your base models (e.g., CNNs for feature extraction)
# Load them the same way you did during training
mobilenet = tf.keras.models.load_model("mobilenet.h5")
vgg16 = tf.keras.models.load_model("vgg16.h5")
efficientnet = tf.keras.models.load_model("efficientnet.h5")
# Add more if needed

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return img_array.reshape(1, 224, 224, 3)

def extract_stacked_features(image):
    image = preprocess_image(image)
    
    p1 = mobilenet.predict(image)[0]
    p2 = vgg16.predict(image)[0]
    p3 = efficientnet.predict(image)[0]
    # Add more predictions if your ensemble used more

    stacked = np.concatenate([p1, p2, p3])  # shape: (num_models × num_classes,)
    return stacked.reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    image = Image.open(file.stream).convert("RGB")
    features = extract_stacked_features(image)

    prediction = model.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
