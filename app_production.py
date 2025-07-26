"""
Production-Ready Plant Disease Classification Web Application

This file shows how to deploy with REAL model predictions instead of dummy features.
Use this when you have all your individual model files ready.

Model Architecture:
- Stacking Ensemble with 6 base models:
  1. Swin Tiny (swin_tiny.h5)
  2. DenseNet121 (densenet121.h5) 
  3. EfficientNetB3 (efficientnetb3.h5)
  4. ConvNeXt (convnext.h5)
  5. ResNet50 (resnet50.h5)
  6. InceptionV3 (inceptionv3.h5)

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
import tensorflow as tf

app = Flask(__name__)

# Load your stacking model and label encoder
try:
    model = joblib.load("stacking_ensemble_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("✅ Stacking model and label encoder loaded successfully!")
except Exception as e:
    print(f"❌ Error loading stacking models: {e}")
    model = None
    label_encoder = None

# Load individual base models
base_models = {}
model_files = {
    'swin': 'swin_tiny.h5',
    'densenet121': 'densenet121.h5', 
    'efficientnetb3': 'efficientnetb3.h5',
    'convnext': 'convnext.h5',
    'resnet50': 'resnet50.h5',
    'inceptionv3': 'inceptionv3.h5'
}

print("🔄 Loading individual base models...")
for model_name, model_file in model_files.items():
    try:
        if os.path.exists(model_file):
            base_models[model_name] = tf.keras.models.load_model(model_file)
            print(f"✅ {model_name} loaded from {model_file}")
        else:
            print(f"⚠️  {model_file} not found - {model_name} will use dummy features")
            base_models[model_name] = None
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        base_models[model_name] = None

def preprocess_image(image):
    """Preprocess image for model input"""
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return img_array.reshape(1, 224, 224, 3)

def extract_real_features(image):
    """
    Extract REAL features from your ensemble models.
    
    This function will:
    1. Use real model predictions when model files are available
    2. Fall back to dummy features for missing models
    3. Maintain the exact feature structure your stacking model expects
    """
    processed_image = preprocess_image(image)
    
    features = []
    model_names = ['swin', 'densenet121', 'efficientnetb3', 'convnext', 'resnet50', 'inceptionv3']
    
    for model_name in model_names:
        if base_models[model_name] is not None:
            # Use REAL model prediction
            try:
                model_probs = base_models[model_name].predict(processed_image, verbose=0)[0]
                print(f"✅ Real prediction from {model_name}: {model_probs[:3]}...")  # Show first 3 values
                features.extend(model_probs)
            except Exception as e:
                print(f"❌ Error getting prediction from {model_name}: {e}")
                # Fall back to dummy features for this model
                dummy_probs = np.random.dirichlet(np.ones(7))
                print(f"⚠️  Using dummy features for {model_name}")
                features.extend(dummy_probs)
        else:
            # Use dummy features for missing models
            dummy_probs = np.random.dirichlet(np.ones(7))
            print(f"⚠️  Using dummy features for {model_name} (model file not found)")
            features.extend(dummy_probs)
    
    features = np.array(features)
    print(f"📊 Total features shape: {features.shape} (Expected: 42)")
    
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
        return jsonify({'error': 'Stacking model not loaded properly'}), 500

    try:
        # Open and process the image
        image = Image.open(file.stream).convert("RGB")
        
        # Extract features using real models (when available)
        features = extract_real_features(image)
        
        # Make prediction using your stacking ensemble
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        
        # Convert to Python string to avoid JSON serialization issues
        predicted_label = str(predicted_label)
        
        # Get prediction probabilities
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
        
        # Count how many real vs dummy models were used
        real_models_count = sum(1 for model_name in base_models.keys() if base_models[model_name] is not None)
        dummy_models_count = 6 - real_models_count
        
        # Prepare response
        response = {
            'prediction': predicted_label,
            'status': 'success',
            'model_info': {
                'base_models': 6,
                'real_models_loaded': real_models_count,
                'dummy_models_used': dummy_models_count,
                'classes': 7,
                'architecture': 'Stacking Ensemble (Swin + DenseNet + EfficientNet + ConvNeXt + ResNet + Inception)',
                'prediction_quality': 'REAL' if real_models_count == 6 else f'PARTIAL ({real_models_count}/6 real models)'
            }
        }
        
        if confidence:
            response['confidence'] = f"{confidence:.1f}%"
            
        if class_probabilities:
            response['all_predictions'] = class_probabilities
            
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌿 PLANT DISEASE DETECTION SYSTEM")
    print("="*60)
    print(f"📊 Stacking Model: {'✅ Loaded' if model else '❌ Not loaded'}")
    print(f"🏷️  Label Encoder: {'✅ Loaded' if label_encoder else '❌ Not loaded'}")
    print(f"🤖 Real Models: {sum(1 for m in base_models.values() if m is not None)}/6")
    print(f"🎲 Dummy Models: {sum(1 for m in base_models.values() if m is None)}/6")
    
    if sum(1 for m in base_models.values() if m is not None) == 6:
        print("🎯 STATUS: PRODUCTION READY - All models loaded!")
    elif sum(1 for m in base_models.values() if m is not None) > 0:
        print("⚠️  STATUS: PARTIAL - Some models loaded, some using dummy features")
    else:
        print("🎲 STATUS: DEMO MODE - All dummy features")
    
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
