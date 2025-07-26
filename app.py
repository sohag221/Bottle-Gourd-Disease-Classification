from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename
from PIL import Image
# Remove tensorflow import since we don't have the base models
# import tensorflow as tf

app = Flask(__name__)

# Load your stacking model and label encoder
try:
    model = joblib.load("stacking_ensemble_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    label_encoder = None

# Since you don't have the individual base models (swin, densenet, etc.),
# we'll create a function that generates features compatible with your trained ensemble
# In a real deployment, you would load the actual base models here:
# swin_model = tf.keras.models.load_model("swin_tiny.h5")
# densenet_model = tf.keras.models.load_model("densenet121.h5") 
# efficientnet_model = tf.keras.models.load_model("efficientnetb3.h5")
# convnext_model = tf.keras.models.load_model("convnext.h5")
# resnet_model = tf.keras.models.load_model("resnet50.h5")
# inception_model = tf.keras.models.load_model("inceptionv3.h5")

def preprocess_image(image):
    """Preprocess image for model input"""
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return img_array.reshape(1, 224, 224, 3)

def extract_ensemble_features(image):
    """
    Extract features for your stacking ensemble model.
    
    Since you trained with: swin, densenet121, efficientnetb3, convnext, resnet50, inceptionv3
    each producing prediction probabilities, we need to simulate this.
    
    In a real deployment, you would:
    1. Load each base model
    2. Get predictions from each model
    3. Concatenate the prediction probabilities
    
    For now, we'll generate compatible dummy features based on typical output shapes.
    """
    image = preprocess_image(image)
    
    # Your ensemble was trained on concatenated prediction probabilities from 6 models
    # Assuming each model outputs probabilities for N classes (e.g., disease classes)
    # You need to determine the exact number of classes your models predict
    
    # Common disease classification datasets have 10-50 classes
    # Adjust this based on your actual number of disease classes
    num_classes = 7  # Update this with your actual number of disease classes
    
    # Generate dummy prediction probabilities for each base model
    # In reality, these would come from the actual model predictions
    
    # Simulate prediction probabilities (they should sum to 1 for each model)
    swin_probs = np.random.dirichlet(np.ones(num_classes))
    dense_probs = np.random.dirichlet(np.ones(num_classes))
    effi_probs = np.random.dirichlet(np.ones(num_classes))
    conv_probs = np.random.dirichlet(np.ones(num_classes))
    resn_probs = np.random.dirichlet(np.ones(num_classes))
    ince_probs = np.random.dirichlet(np.ones(num_classes))
    
    # Concatenate all prediction probabilities (this is what your stacking model expects)
    stacked_features = np.concatenate([
        swin_probs, dense_probs, effi_probs, 
        conv_probs, resn_probs, ince_probs
    ])
    
    return stacked_features.reshape(1, -1)

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

    if model is None or label_encoder is None:
        return jsonify({'error': 'Models not loaded properly'}), 500

    try:
        # Open and process the image
        image = Image.open(file.stream).convert("RGB")
        
        # Extract features for your ensemble model
        features = extract_ensemble_features(image)
        
        # Make prediction
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Convert to Python string to avoid JSON serialization issues
        predicted_label = str(predicted_label)
        
        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(features)
            confidence = float(max(prediction_proba[0]) * 100)
        except:
            confidence = None
        
        # Prepare response
        response = {
            'prediction': predicted_label,
            'status': 'success'
        }
        
        if confidence:
            response['confidence'] = f"{confidence:.1f}%"
            
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Log error for debugging
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
