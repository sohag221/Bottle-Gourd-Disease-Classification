# 🌿 Plant Disease Detection System

A powerful AI-driven plant disease detection system using PyTorch ensemble models for accurate identification of plant diseases from leaf and fruit images.

## 🚀 Features

- **6 PyTorch Model Ensemble**: ResNet50, EfficientNetB3, DenseNet121, InceptionV3, ConvNeXt, and Swin Tiny
- **Real-time Prediction**: Upload images and get instant disease classification
- **7 Disease Classes**: Comprehensive detection of common plant diseases
- **Interactive Web Interface**: User-friendly Flask-based web application
- **Confidence Scoring**: Detailed probability breakdown for all disease classes
- **Individual Model Results**: See predictions from each model in the ensemble

## 📊 Supported Disease Classes

1. **Anthracnose fruit rot**
2. **Anthracnose leaf spot**
3. **Blossom end rot**
4. **Fresh fruit** (Healthy)
5. **Fresh leaf** (Healthy)
6. **Insect damaged leaf**
7. **Yellow mosaic virus**

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: PyTorch, timm
- **Image Processing**: PIL, torchvision
- **Frontend**: HTML5, CSS3, JavaScript
- **Model Storage**: Git LFS for large model files

## 📋 Prerequisites

- Python 3.8 or higher
- Git LFS (for downloading model files)
- Virtual environment (recommended)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sohag221/leaf_disease_app.git
cd leaf_disease_app
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install flask pillow numpy scikit-learn timm
```

### 4. Download Model Files

The PyTorch model files are stored using Git LFS. Make sure you have Git LFS installed:

```bash
# Install Git LFS if not already installed
git lfs install

# Download model files
git lfs pull
```

## 🚀 Usage

### Running the Application

1. **Start the Flask Server**:
```bash
python app.py
```

2. **Access the Web Interface**:
   - Open your browser and go to `http://localhost:5000`
   - Or access via network: `http://your-ip:5000`

3. **Upload and Predict**:
   - Click "Choose Image" to upload a leaf or fruit image
   - Click "Predict Disease" to get analysis results
   - View ensemble prediction and individual model results

### API Usage

You can also use the prediction API directly:

```python
import requests

# Upload image for prediction
url = "http://localhost:5000/predict"
files = {"image": open("path/to/your/image.jpg", "rb")}
response = requests.post(url, files=files)
result = response.json()

print(f"Predicted Disease: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

## 🏗️ Project Structure

```
leaf_disease_app/
├── app.py                          # Main Flask application
├── templates/
│   └── index.html                  # Web interface template
├── static/
│   └── style.css                   # Styling for web interface
├── *.pth                          # PyTorch model files (via Git LFS)
├── label_encoder.pkl              # Label encoder for class mapping
├── .gitattributes                 # Git LFS configuration
└── README.md                      # This file
```

## 🧠 Model Architecture

### Ensemble Approach
The system uses an ensemble of 6 different CNN architectures:

1. **ResNet50**: Deep residual network with skip connections
2. **EfficientNetB3**: Efficient scaling of CNN architectures
3. **DenseNet121**: Dense connectivity between layers
4. **InceptionV3**: Multi-scale feature extraction
5. **ConvNeXt**: Modern CNN architecture
6. **Swin Tiny**: Vision transformer with shifted windows

### Prediction Process

1. **Image Preprocessing**: Resize to 224×224, normalize with ImageNet standards
2. **Individual Predictions**: Each model generates probability scores
3. **Ensemble Averaging**: Average probabilities across all models
4. **Final Classification**: Select class with highest ensemble probability

## 🔍 How It Works

### Image Processing Pipeline

```python
# 1. Image Upload and Validation
image = Image.open(uploaded_file).convert("RGB")

# 2. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 3. Model Inference
for model_name, model in models.items():
    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1)

# 4. Ensemble Prediction
ensemble_probs = np.mean(all_predictions, axis=0)
final_prediction = np.argmax(ensemble_probs)
```

## 📈 Performance

- **Ensemble Accuracy**: Improved accuracy through model averaging
- **Individual Model Insights**: View contribution of each architecture
- **Confidence Scoring**: Reliable confidence estimates
- **Real-time Processing**: Fast inference on CPU/GPU

## 🔬 Model Details

### Input Requirements
- **Image Format**: JPG, PNG, or other common formats
- **Recommended Size**: Any size (automatically resized to 224×224)
- **Color Space**: RGB (automatically converted if needed)

### Output Format
```json
{
  "prediction": "Anthracnose leaf spot",
  "confidence": "92.45%",
  "all_predictions": {
    "Anthracnose leaf spot": "92.45%",
    "Fresh leaf": "4.32%",
    "Insect damaged leaf": "2.11%",
    "..."
  },
  "individual_models": {
    "resnet50": {"prediction": 1, "confidence": "89.23%"},
    "efficientnetb3": {"prediction": 1, "confidence": "94.12%"},
    "..."
  },
  "model_info": {
    "architecture": "PyTorch Ensemble",
    "total_models": 6,
    "device": "cpu"
  }
}
```

## 🛡️ Error Handling

The application includes comprehensive error handling:

- **File Validation**: Ensures valid image files are uploaded
- **Model Loading**: Graceful handling of missing model files
- **Prediction Errors**: Detailed error messages for debugging
- **Network Issues**: Timeout and connection error handling

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment

For production deployment, consider using:

- **Gunicorn**: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
- **Docker**: Containerize the application
- **Cloud Services**: Deploy on AWS, GCP, or Azure
- **Load Balancer**: For high-traffic scenarios

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **timm Library**: For pre-trained model architectures
- **Flask Community**: For the lightweight web framework
- **Open Source Community**: For continuous inspiration and support

## 📞 Contact

- **GitHub**: [@sohag221](https://github.com/sohag221)
- **Repository**: [leaf_disease_app](https://github.com/sohag221/leaf_disease_app)

## 🐛 Issues

If you encounter any issues or have suggestions, please create an issue on GitHub:
[Create Issue](https://github.com/sohag221/leaf_disease_app/issues)

---

**Made with ❤️ for the agricultural community**
