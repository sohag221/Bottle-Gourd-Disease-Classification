import gradio as gr
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

print("🔄 Loading PyTorch models...")
for model_name, model_file in model_files.items():
    try:
        model = create_model(model_name, NUM_CLASSES)
        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        models[model_name] = model
        print(f"✅ Loaded {model_name}")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {str(e)}")

print(f"📊 Successfully loaded {len(models)} models")

# Load label encoder
try:
    label_encoder = joblib.load('label_encoder.pkl')
    print(f"✅ Loaded label encoder with {len(label_encoder.classes_)} classes")
except Exception as e:
    print(f"❌ Failed to load label encoder: {str(e)}")
    label_encoder = None

# Disease name mapping
def get_disease_name(class_index):
    """Convert class index to readable disease name"""
    disease_names = {
        0: "Anthracnose fruit rot",
        1: "Anthracnose leaf spot", 
        2: "Blossom end rot",
        3: "Fresh fruit",
        4: "Fresh leaf",
        5: "Insect damaged leaf",
        6: "Yellow mosaic virus"
    }
    return disease_names.get(class_index, f"Class {class_index}")

def ensemble_predict(image):
    """Make predictions using all loaded models and return ensemble result"""
    try:
        # Preprocess image
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Get predictions from all models
        all_predictions = []
        model_names = []
        
        with torch.no_grad():
            for model_name, model in models.items():
                try:
                    outputs = model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    all_predictions.append(probabilities.cpu().numpy())
                    model_names.append(model_name)
                except Exception as e:
                    print(f"Error with {model_name}: {str(e)}")
                    continue
        
        if not all_predictions:
            return "Error: No models could make predictions", {}, {}
        
        # Ensemble averaging
        ensemble_probs = np.mean(all_predictions, axis=0)[0]
        predicted_class = np.argmax(ensemble_probs)
        confidence = float(ensemble_probs[predicted_class])
        
        # Get disease name
        disease_name = get_disease_name(predicted_class)
        
        # Prepare all class probabilities
        all_class_probs = {}
        for i, prob in enumerate(ensemble_probs):
            class_name = get_disease_name(i)
            all_class_probs[class_name] = f"{prob:.2%}"
        
        # Sort by probability (highest first)
        all_class_probs = dict(sorted(all_class_probs.items(), 
                                    key=lambda x: float(x[1][:-1]), reverse=True))
        
        # Model info
        model_info = {
            "Architecture": "Ensemble of 6 CNN Models",
            "Base Models": model_names,
            "Total Models": len(model_names),
            "Classes": NUM_CLASSES,
            "Device": str(DEVICE)
        }
        
        return disease_name, confidence, all_class_probs, model_info
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg, 0.0, {}, {}

def predict_disease(image):
    """Main prediction function for Gradio interface"""
    if image is None:
        return "Please upload an image", "", ""
    
    try:
        # Make prediction
        disease_name, confidence, all_probs, model_info = ensemble_predict(image)
        
        # Format main result
        main_result = f"🔬 **Detected Disease:** {disease_name}\n"
        main_result += f"📊 **Confidence:** {confidence:.2%}\n\n"
        
        # Format all probabilities
        prob_result = "📈 **All Class Probabilities:**\n"
        for disease, prob in all_probs.items():
            prob_result += f"• {disease}: {prob}\n"
        
        # Format model info
        info_result = "🤖 **Model Information:**\n"
        info_result += f"• Architecture: {model_info.get('Architecture', 'N/A')}\n"
        info_result += f"• Base Models: {', '.join(model_info.get('Base Models', []))}\n"
        info_result += f"• Total Models: {model_info.get('Total Models', 0)}\n"
        info_result += f"• Device: {model_info.get('Device', 'N/A')}\n"
        
        return main_result, prob_result, info_result
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "", ""

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="🌿 Bottle Gourd Disease Detection System",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; color: #2d5a27; }
        .disease-tag { 
            background: #e8f5e8; 
            padding: 5px 10px; 
            border-radius: 15px; 
            color: #27ae60; 
            margin: 2px;
            display: inline-block;
        }
        """
    ) as interface:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🌿 Bottle Gourd Disease Detection System</h1>
            <p>Upload a leaf or fruit image to detect plant diseases using AI ensemble models</p>
        </div>
        """)
        
        gr.HTML("""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 20px 0;">
            <h4 style="color: #2c3e50; margin-bottom: 10px;">🔬 Detectable Diseases:</h4>
            <div>
                <span class="disease-tag">Anthracnose fruit rot</span>
                <span class="disease-tag">Anthracnose leaf spot</span>
                <span class="disease-tag">Blossom end rot</span>
                <span class="disease-tag">Fresh fruit</span>
                <span class="disease-tag">Fresh leaf</span>
                <span class="disease-tag">Insect damaged leaf</span>
                <span class="disease-tag">Yellow mosaic virus</span>
            </div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📁 Upload Plant Image",
                    type="pil",
                    height=400
                )
                
                predict_btn = gr.Button(
                    "🔍 Analyze Disease",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                main_output = gr.Markdown(
                    label="🎯 Main Result",
                    value="Upload an image to see the analysis results..."
                )
                
                prob_output = gr.Markdown(
                    label="📊 Detailed Probabilities"
                )
                
                model_output = gr.Markdown(
                    label="🤖 Model Information"
                )
        
        # Set up the prediction function
        predict_btn.click(
            fn=predict_disease,
            inputs=[image_input],
            outputs=[main_output, prob_output, model_output]
        )
        
        # Also trigger on image upload
        image_input.change(
            fn=predict_disease,
            inputs=[image_input],
            outputs=[main_output, prob_output, model_output]
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; color: #7f8c8d;">
            <p>🤖 Powered by PyTorch Ensemble Models | 🚀 Deployed on Hugging Face Spaces</p>
        </div>
        """)
    
    return interface

# Launch the interface
if __name__ == "__main__":
    print("=" * 50)
    print("🌿 BOTTLE GOURD DISEASE DETECTION SYSTEM")
    print("=" * 50)
    print(f"🔧 Device: {DEVICE}")
    print(f"🎯 Model architectures: {list(models.keys())}")
    if label_encoder is not None:
        print(f"📊 Disease classes: {len(label_encoder.classes_)}")
        print(f"🔬 Detectable diseases: {list(label_encoder.classes_)}")
    print("🌐 Starting Gradio interface...")
    print("=" * 50)
    
    interface = create_interface()
    interface.launch(share=True)
