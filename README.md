# 🌿 Plant Disease Detection System

An AI-powered web application that detects plant diseases from leaf images using an ensemble of 6 deep learning models. Share your app globally with friends anywhere in the world!

![Plant Disease Detection](https://img.shields.io/badge/AI-Plant%20Disease%20Detection-green)
![Python](https://img.shields.io/badge/Python-3.13.5-blue)
![Flask](https://img.shields.io/badge/Flask-3.1.1-red)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-orange)

## 🚀 Features

- **6 PyTorch Model Ensemble**: ResNet50, EfficientNetB3, DenseNet121, InceptionV3, ConvNeXt, and Swin Tiny
- **Real-time Prediction**: Upload images and get instant disease classification
- **7 Disease Classes**: Comprehensive detection of common plant diseases
- **Interactive Web Interface**: User-friendly Flask-based web application with professional UI
- **Confidence Scoring**: Detailed probability breakdown for all disease classes
- **Individual Model Results**: See predictions from each model in the ensemble
- **Global Sharing**: Share your app worldwide using ngrok tunneling service
- **Mobile Friendly**: Responsive design works on any device (phone, computer, tablet)
- **Secure Access**: HTTPS encryption for global connections

## 🌐 Quick Global Sharing

### Step 1: Start Your App
```bash
# Start Flask application
D:/leaf_disease_app/.venv/Scripts/python.exe app.py
```

### Step 2: Create Global Tunnel
```bash
# Set up ngrok (one-time setup)
ngrok authtoken YOUR_TOKEN_HERE

# Create global tunnel
ngrok http 5000
```

### Step 3: Share the URL
- Copy the HTTPS URL (e.g., `https://abc123.ngrok-free.app`)
- Share with friends anywhere in the world
- They can access your app instantly on any device

## 📊 Supported Disease Classes

1. **Anthracnose fruit rot**
2. **Anthracnose leaf spot**
3. **Blossom end rot**
4. **Fresh fruit** (Healthy)
5. **Fresh leaf** (Healthy)
6. **Insect damaged leaf**
7. **Yellow mosaic virus**

## 🛠️ Technology Stack

- **Backend**: Flask 3.1.1 (Python 3.13.5)
- **Machine Learning**: PyTorch 2.7.1, timm
- **Image Processing**: PIL 10.4.0, torchvision 0.22.1
- **Frontend**: HTML5, CSS3, JavaScript (Professional UI)
- **Global Sharing**: ngrok tunneling service
- **Environment**: Virtual environment (.venv) configured

## 📋 Prerequisites

- Python 3.13.5 (configured)
- Virtual environment (.venv) - already set up
- ngrok (for global sharing)
- All dependencies installed

## 🔧 Installation & Setup

### Ready to Use! 
Your environment is already configured with:
- ✅ Python 3.13.5 virtual environment
- ✅ All required dependencies installed
- ✅ PyTorch 2.7.1 with CPU support
- ✅ Model files ready

### For Global Sharing Setup

1. **Install/Update ngrok:**
   ```bash
   # Download from https://ngrok.com/download
   # Ensure version 3.7.0+ 
   ngrok update
   ```

2. **Get ngrok Auth Token:**
   ```bash
   # Sign up at https://ngrok.com
   # Get token from dashboard
   ngrok authtoken YOUR_TOKEN_HERE
   ```

## 🚀 Usage

### Local Usage
```bash
# Start the application
D:/leaf_disease_app/.venv/Scripts/python.exe app.py

# Access at: http://localhost:5000
```

### Global Sharing (Recommended)
```bash
# Terminal 1: Start Flask app
D:/leaf_disease_app/.venv/Scripts/python.exe app.py

# Terminal 2: Create global tunnel  
ngrok http 5000

# Share the HTTPS URL with friends worldwide!
```

### Alternative Sharing Methods
```bash
# Method 1: Automated script
python share_app.py

# Method 2: Batch script (Windows)
share_globally.bat
```

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
### Web Interface Usage

1. **Local Access**: Open `http://localhost:5000`
2. **Global Access**: Use ngrok URL (e.g., `https://abc123.ngrok-free.app`)
3. **Upload Image**: Click "Choose File" and select plant leaf image
4. **Analyze**: Click "Analyze Image" to get disease detection results
5. **View Results**: See ensemble prediction with confidence scores

### API Usage

You can also use the prediction API directly:

```python
import requests

# For local usage
url = "http://localhost:5000/predict"

# For global usage (replace with your ngrok URL)
url = "https://abc123.ngrok-free.app/predict"

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
├── utils.py                        # Utility functions
├── share_app.py                    # Global sharing script
├── setup_ngrok.py                  # ngrok setup helper
├── share_globally.bat              # Windows batch script
├── templates/
│   └── index.html                  # Professional web interface
├── static/
│   └── style.css                   # Modern UI styling
├── stacking_ensemble_model.pkl     # Trained ensemble model
├── label_encoder.pkl               # Label encoder for class mapping
├── .venv/                          # Python virtual environment
├── README.md                       # Complete documentation
├── SHARING_GUIDE.md               # Detailed sharing guide
└── .gitattributes                  # Git LFS configuration
```

## 🔗 Sharing Files Created

- `share_app.py`: Automated Python script for global sharing
- `setup_ngrok.py`: Interactive ngrok authentication setup
- `share_globally.bat`: Windows batch script for easy sharing
- `SHARING_GUIDE.md`: Comprehensive sharing documentation

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

## 📈 Performance & Global Sharing

### Model Performance
- **Ensemble Accuracy**: Improved accuracy through 6-model averaging
- **Individual Model Insights**: View contribution of each architecture
- **Confidence Scoring**: Reliable confidence estimates
- **Real-time Processing**: Fast inference on CPU (optimized for deployment)

### Global Sharing Performance
- **Connection Speed**: Instant global access via ngrok tunneling
- **Device Compatibility**: Works on any device (phone, computer, tablet)
- **Network Reliability**: HTTPS secure connections worldwide
- **Session Duration**: Free ngrok sessions last 2 hours

## 🌍 Current Deployment Status

### Active Global URL
Your app is currently accessible worldwide at:
```
https://420ecffa161d.ngrok-free.app
```

### Sharing Capabilities
- ✅ **Global Access**: Available from any country
- ✅ **Mobile Friendly**: Responsive design for all devices
- ✅ **Secure Connection**: HTTPS encryption
- ✅ **Real-time Processing**: Instant disease detection
- ✅ **No Installation**: Friends just need the URL

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

## 🛡️ Error Handling & Security

### Application Security
- **File Validation**: Ensures valid image files are uploaded
- **Model Loading**: Graceful handling of missing model files
- **Prediction Errors**: Detailed error messages for debugging
- **Network Issues**: Timeout and connection error handling

### Global Sharing Security
- **HTTPS Encryption**: All global connections are secure
- **Private IP Protection**: Your computer's IP remains hidden
- **Temporary Access**: ngrok URLs expire when you stop sharing
- **No Data Storage**: Images are processed in real-time only

## 🚀 Deployment Options

### 1. Local Development (Current)
```bash
D:/leaf_disease_app/.venv/Scripts/python.exe app.py
# Access: http://localhost:5000
```

### 2. Global Sharing (Active)
```bash
# Terminal 1: Flask app
D:/leaf_disease_app/.venv/Scripts/python.exe app.py

# Terminal 2: ngrok tunnel
ngrok http 5000
# Access: https://420ecffa161d.ngrok-free.app
```

### 3. Automated Sharing
```bash
python share_app.py
# Automatically handles Flask + ngrok setup
```

### 4. Production Deployment (Future)
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

## 📄 Documentation Files

- `README.md`: Complete project documentation (this file)
- `SHARING_GUIDE.md`: Detailed global sharing guide
- Project includes all necessary sharing scripts and setup files

## 📝 License

This project is developed for educational and research purposes in plant disease detection using deep learning.

---

**🌟 Share your AI-powered plant disease detection system with the world! 🌍**

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
