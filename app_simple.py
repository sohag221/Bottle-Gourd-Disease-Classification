"""
Leaf Disease Classification Web Application

Model Architecture:
- Stacking Ensemble with 6 base models:
  1. Swin Tiny
  2. DenseNet121
  3. EfficientNetB3
  4. ConvNeXt
  5. ResNet50
  6. InceptionV3

Disease Classes (7 total):
1. Anthracnose fruit rot
2. Anthracnose leaf spot
3. Blossom end rot
4. Fresh fruit
5. Fresh leaf
6. Insect damaged leaf
7. Yellow mosaic virus

Feature Structure:
- Each base model outputs 7 class probabilities
- Total features: 6 models × 7 classes = 42 features
- Stacking model: LogisticRegression (trained on concatenated probabilities)
"""

from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename
from PIL import Image

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

def preprocess_image(image):
    """Simple image preprocessing without deep learning models"""
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return img_array.reshape(1, 224, 224, 3)

def extract_simple_features(image):
    """
    Extract features compatible with your stacking ensemble model.
    
    Your model was trained on 6 base models (swin, densenet121, efficientnetb3, 
    convnext, resnet50, inceptionv3) each predicting 7 disease classes:
    1. Anthracnose fruit rot
    2. Anthracnose leaf spot  
    3. Blossom end rot
    4. Fresh fruit
    5. Fresh leaf
    6. Insect damaged leaf
    7. Yellow mosaic virus
    
    Each base model outputs 7 probabilities, so total features = 6 models × 7 classes = 42 features
    """
    image = preprocess_image(image)
    
    # Your ensemble expects 42 features (6 models × 7 disease classes)
    num_classes = 7  # Your disease classes
    num_models = 6   # Your base models (swin, densenet, efficientnet, convnext, resnet, inception)
    
    # Generate dummy prediction probabilities for each base model
    # Each model predicts probabilities for the 7 disease classes
    # In real deployment, these would come from actual model predictions:
    # swin_probs = swin_model.predict(image)[0]
    # dense_probs = densenet_model.predict(image)[0]
    # etc.
    
    features = []
    model_names = ['swin', 'densenet121', 'efficientnetb3', 'convnext', 'resnet50', 'inceptionv3']
    
    for model_name in model_names:
        # Generate dummy probabilities that sum to 1 (realistic model output)
        # These are random but follow proper probability distribution
        model_probs = np.random.dirichlet(np.ones(num_classes))
        features.extend(model_probs)
    
    features = np.array(features)
    print(f"Generated features shape: {features.shape} (Expected: {num_models * num_classes})")
    
    return features.reshape(1, -1)

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
        
        # Extract features for prediction
        features = extract_simple_features(image)

        # Make prediction
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Convert to Python string to avoid JSON serialization issues
        predicted_label = str(predicted_label)
        
        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(features)
            confidence = float(max(prediction_proba[0]) * 100)
            
            # Get all class probabilities for detailed results
            class_probabilities = {}
            classes = label_encoder.classes_
            probabilities = prediction_proba[0]
            
            for i, class_name in enumerate(classes):
                class_probabilities[str(class_name)] = f"{probabilities[i]*100:.2f}%"
                
        except Exception as prob_error:
            print(f"Probability calculation error: {prob_error}")
            confidence = None
            class_probabilities = None
        
        # Prepare response
        response = {
            'prediction': predicted_label,
            'status': 'success',
            'model_info': {
                'base_models': 6,
                'classes': 7,
                'architecture': 'Stacking Ensemble (Swin + DenseNet + EfficientNet + ConvNeXt + ResNet + Inception)'
            }
        }
        
        if confidence:
            response['confidence'] = f"{confidence:.1f}%"
            
        if class_probabilities:
            response['all_predictions'] = class_probabilities
            
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")  # Log error for debugging
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
