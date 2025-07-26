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
    """Extract simple features from image for demonstration"""
    image = preprocess_image(image)
    
    # For demo purposes, we'll create dummy features
    # In a real scenario, you would have the actual base model predictions
    # This is just to make the app work without TensorFlow models
    
    # Create some dummy features based on image statistics
    mean_rgb = np.mean(image, axis=(1, 2))  # Mean RGB values
    std_rgb = np.std(image, axis=(1, 2))    # Standard deviation of RGB values
    
    # Create a feature vector (adjust size based on your actual trained model)
    # You'll need to replace this with actual features your model expects
    dummy_features = np.concatenate([mean_rgb.flatten(), std_rgb.flatten()])
    
    # The model expects exactly 42 features based on the error message
    expected_size = 42  # Adjusted based on the error message
    if len(dummy_features) < expected_size:
        # Pad with zeros if we have fewer features
        dummy_features = np.pad(dummy_features, (0, expected_size - len(dummy_features)))
    else:
        # Trim to exact size if we have more features
        dummy_features = dummy_features[:expected_size]
    
    return dummy_features.reshape(1, -1)

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
