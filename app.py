from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
from PIL import Image
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from pathlib import Path

app = Flask(__name__)

# Define the number of classes (adjust based on your dataset)
NUM_CLASSES = 7

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {DEVICE}")

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model definitions
def create_model(model_name, num_classes):
    """Create a model based on the architecture name"""
    if model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
    elif model_name == 'efficientnetb3':
        model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)
    elif model_name == 'densenet121':
        model = timm.create_model('densenet121', pretrained=False, num_classes=num_classes)
    elif model_name == 'inceptionv3':
        model = timm.create_model('inception_v3', pretrained=False, num_classes=num_classes)
    elif model_name == 'convnext':
        model = timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes)
    elif model_name == 'swin_tiny':
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# Load all models
models = {}
model_files = {
    'resnet50': 'resnet50_best.pth',
    'efficientnetb3': 'efficientnetb3_best.pth', 
    'densenet121': 'densenet121_best.pth',
    'inceptionv3': 'inceptionv3_best.pth',
    'convnext': 'convnext_best.pth',
    'swin_tiny': 'swin_tiny_best.pth'
}

print("🚀 Loading PyTorch models...")
for model_name, file_path in model_files.items():
    try:
        if Path(file_path).exists():
            model = create_model(model_name, NUM_CLASSES)
            
            # Load the state dict
            checkpoint = torch.load(file_path, map_location=DEVICE)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(DEVICE)
            model.eval()
            models[model_name] = model
            print(f"✅ Loaded {model_name}")
        else:
            print(f"❌ Model file not found: {file_path}")
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")

# Load label encoder
try:
    label_encoder = joblib.load("label_encoder.pkl")
    print(f"✅ Label encoder loaded: {label_encoder.classes_}")
except Exception as e:
    print(f"❌ Error loading label encoder: {e}")
    # Create a default label encoder if not found
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array([f'Disease_{i}' for i in range(NUM_CLASSES)])
    print(f"🔧 Using default classes: {label_encoder.classes_}")

# Define proper disease class names that match your frontend
disease_class_names = [
    "Anthracnose fruit rot",
    "Anthracnose leaf spot", 
    "Blossom end rot",
    "Fresh fruit",
    "Fresh leaf",
    "Insect damaged leaf",
    "Yellow mosaic virus"
]

# Create a mapping function to convert indices to disease names
def get_disease_name(class_index):
    """Convert class index to disease name"""
    if 0 <= class_index < len(disease_class_names):
        return disease_class_names[class_index]
    return f"Unknown_Disease_{class_index}"

print(f"📊 Total models loaded: {len(models)}")
print(f"🎯 Disease classes: {len(label_encoder.classes_)}")

def preprocess_image(image):
    """Preprocess image for PyTorch models"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(DEVICE)
    
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        raise

def ensemble_predict(image_tensor):
    """Make predictions using all loaded models"""
    predictions = []
    model_results = {}
    
    with torch.no_grad():
        for model_name, model in models.items():
            try:
                # Get model prediction
                outputs = model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predictions.append(probabilities.cpu().numpy())
                
                # Store individual model results
                probs = probabilities.cpu().numpy()[0]
                predicted_class = np.argmax(probs)
                confidence = float(probs[predicted_class] * 100)
                
                model_results[model_name] = {
                    'prediction': int(predicted_class),
                    'confidence': f"{confidence:.2f}%",
                    'probabilities': probs.tolist()
                }
                
                print(f"🔍 {model_name}: Class {predicted_class} ({confidence:.2f}%)")
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                continue
    
    if not predictions:
        raise Exception("No models could make predictions")
    
    # Ensemble prediction (average probabilities)
    ensemble_probs = np.mean(predictions, axis=0)[0]
    ensemble_prediction = np.argmax(ensemble_probs)
    ensemble_confidence = float(ensemble_probs[ensemble_prediction] * 100)
    
    return ensemble_prediction, ensemble_probs, model_results, ensemble_confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate request
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if models are loaded
        if not models:
            return jsonify({'error': 'No models loaded properly'}), 500
        
        # Process the image
        print(f"🖼️ Processing image: {file.filename}")
        image = Image.open(file.stream).convert("RGB")
        print(f"📏 Image size: {image.size}")
        
        # Preprocess image for PyTorch models
        image_tensor = preprocess_image(image)
        print(f"🔢 Tensor shape: {image_tensor.shape}")
        
        # Make ensemble predictions
        ensemble_prediction, ensemble_probs, model_results, ensemble_confidence = ensemble_predict(image_tensor)
        
        # Get the predicted class name using our disease name mapping
        predicted_class = get_disease_name(ensemble_prediction)
        
        # Get all class probabilities for ensemble using disease names
        all_predictions = {}
        for i in range(len(disease_class_names)):
            disease_name = get_disease_name(i)
            prob_percentage = float(ensemble_probs[i] * 100)
            all_predictions[disease_name] = f"{prob_percentage:.2f}%"
        
        # Sort predictions by probability (highest first)
        sorted_predictions = dict(sorted(all_predictions.items(), 
                                       key=lambda x: float(x[1].rstrip('%')), 
                                       reverse=True))
        
        print(f"🎯 Ensemble Prediction: {predicted_class} ({ensemble_confidence:.2f}%)")
        
        # Prepare response
        response = {
            'prediction': str(predicted_class),
            'confidence': f"{ensemble_confidence:.2f}%",
            'all_predictions': sorted_predictions,
            'individual_models': model_results,
            'model_info': {
                'architecture': 'PyTorch Ensemble',
                'base_models': list(models.keys()),
                'total_models': len(models),
                'device': str(DEVICE),
                'classes': len(label_encoder.classes_)
            },
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("🌿 Plant Disease Detection System")
    print("=" * 50)
    print("🚀 Starting Flask application...")
    print(f"✅ PyTorch models loaded: {len(models)}")
    print(f"🔧 Device: {DEVICE}")
    print(f"🎯 Model architectures: {list(models.keys())}")
    if label_encoder is not None:
        print(f"📊 Disease classes: {len(label_encoder.classes_)}")
        print(f"🔬 Detectable diseases: {list(label_encoder.classes_)}")
    print("🌐 Access at: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
