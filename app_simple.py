"""
Simple Gradio interface for Bottle Gourd Disease Detection
This version uses simpler components to avoid JSON schema issues
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import timm
import joblib
import gradio as gr

print("🔧 Using device:", "cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disease mapping
DISEASE_NAMES = {
    0: 'Anthracnose',
    1: 'Bacterial leaf spot',
    2: 'Downy mildew',
    3: 'Healthy leaf',
    4: 'Leaf spot',
    5: 'Mosaic virus',
    6: 'Powdery mildew'
}

# Model architectures mapping
MODEL_ARCHITECTURES = {
    'resnet50': 'resnet50',
    'efficientnetb3': 'efficientnet_b3',
    'densenet121': 'densenet121', 
    'inceptionv3': 'inception_v3',
    'convnext': 'convnext_tiny',
    'swin_tiny': 'swin_tiny_patch4_window7_224'
}

def load_model(model_name, num_classes=7):
    """Load a single PyTorch model"""
    try:
        architecture = MODEL_ARCHITECTURES[model_name]
        model = timm.create_model(architecture, pretrained=False, num_classes=num_classes)
        
        model_path = f"{model_name}_best.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        model.to(device)
        print(f"✅ Loaded {model_name}")
        return model
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None

# Load all models
print("🔄 Loading PyTorch models...")
models = {}
model_names = ['resnet50', 'efficientnetb3', 'densenet121', 'inceptionv3', 'convnext', 'swin_tiny']

for name in model_names:
    model = load_model(name)
    if model is not None:
        models[name] = model

print(f"📊 Successfully loaded {len(models)} models")

# Load label encoder
try:
    label_encoder = joblib.load('label_encoder.pkl')
    print(f"✅ Loaded label encoder with {len(label_encoder.classes_)} classes")
except Exception as e:
    print(f"❌ Error loading label encoder: {e}")
    sys.exit(1)

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif hasattr(image, 'convert'):
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0).to(device)

def ensemble_predict(image):
    """Make prediction using ensemble of models"""
    try:
        if image is None:
            return "❌ No image provided"
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Get predictions from all models
        predictions = []
        with torch.no_grad():
            for name, model in models.items():
                output = model(processed_image)
                probabilities = torch.softmax(output, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        if not predictions:
            return "❌ No models available for prediction"
        
        # Average predictions
        avg_prediction = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_prediction)
        confidence = np.max(avg_prediction) * 100
        
        # Get disease name
        disease_name = DISEASE_NAMES.get(predicted_class, f"Unknown (Class {predicted_class})")
        
        # Format result
        result = f"""
🔬 **Prediction Results**

**Disease:** {disease_name}
**Confidence:** {confidence:.1f}%
**Models Used:** {len(models)} ensemble models

📊 **Model Performance:**
"""
        
        for i, (name, prob) in enumerate(zip(models.keys(), predictions)):
            class_pred = np.argmax(prob)
            conf = np.max(prob) * 100
            result += f"\n• {name}: {DISEASE_NAMES.get(class_pred, 'Unknown')} ({conf:.1f}%)"
        
        return result
        
    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"

# Create Gradio interface
def create_simple_interface():
    """Create a simple Gradio interface"""
    
    with gr.Blocks(title="🌿 Bottle Gourd Disease Detection", theme=gr.themes.Soft()) as interface:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🌿 Bottle Gourd Disease Detection System</h1>
            <p>Upload an image of a bottle gourd leaf to detect diseases using our AI ensemble model</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📸 Upload Leaf Image",
                    type="pil"
                )
                predict_btn = gr.Button("🔬 Analyze Disease", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                result_output = gr.Textbox(
                    label="📋 Diagnosis Results",
                    lines=15,
                    max_lines=20
                )
        
        # Connect the prediction function
        predict_btn.click(
            fn=ensemble_predict,
            inputs=image_input,
            outputs=result_output
        )
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 20px; background-color: #f0f0f0; border-radius: 10px;">
            <h3>📚 Detectable Diseases</h3>
            <p><strong>Anthracnose</strong> • <strong>Bacterial Leaf Spot</strong> • <strong>Downy Mildew</strong> • <strong>Healthy Leaf</strong> • <strong>Leaf Spot</strong> • <strong>Mosaic Virus</strong> • <strong>Powdery Mildew</strong></p>
            <p style="margin-top: 15px; font-size: 0.9em; color: #666;">
                🎯 Ensemble of 6 deep learning models • 📊 High accuracy disease detection • 🔬 Research-grade AI system
            </p>
        </div>
        """)
    
    return interface

if __name__ == "__main__":
    print("="*50)
    print("🌿 BOTTLE GOURD DISEASE DETECTION SYSTEM")
    print("="*50)
    print(f"🔧 Device: {device}")
    print(f"🎯 Model architectures: {list(models.keys())}")
    print(f"📊 Disease classes: {len(DISEASE_NAMES)}")
    print(f"🔬 Detectable diseases: {list(DISEASE_NAMES.values())}")
    print("🌐 Starting Gradio interface...")
    print("="*50)
    
    # Create and launch interface
    app = create_simple_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=True,  # Enable public sharing
        show_error=True,
        quiet=False
    )
