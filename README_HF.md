---
title: Bottle Gourd Disease Detection System
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app_hf.py
pinned: false
license: mit
---

# 🌿 Bottle Gourd Disease Detection System

An AI-powered plant disease detection system using ensemble of 6 state-of-the-art CNN models to identify diseases in bottle gourd plants.

## 🎯 Features

- **Multi-Model Ensemble**: Combines predictions from 6 different CNN architectures
- **High Accuracy**: Ensemble approach for improved prediction reliability
- **Real-time Analysis**: Instant disease detection from uploaded images
- **7 Disease Classes**: Detects various bottle gourd diseases and healthy conditions

## 🔬 Detectable Conditions

1. **Anthracnose fruit rot**
2. **Anthracnose leaf spot**
3. **Blossom end rot**
4. **Fresh fruit** (Healthy)
5. **Fresh leaf** (Healthy)
6. **Insect damaged leaf**
7. **Yellow mosaic virus**

## 🤖 Model Architecture

The system uses an ensemble of 6 CNN models:

- **ResNet50**: Deep residual networks for robust feature extraction
- **EfficientNetB3**: Efficient scaling for optimal performance
- **DenseNet121**: Dense connections for feature reuse
- **InceptionV3**: Multi-scale feature processing
- **ConvNeXt Tiny**: Modern ConvNet architecture
- **Swin Tiny**: Vision transformer with shifted windows

## 📊 How It Works

1. **Upload Image**: Select a clear image of bottle gourd leaf or fruit
2. **AI Analysis**: 6 models analyze the image simultaneously
3. **Ensemble Prediction**: Results are combined for final prediction
4. **Detailed Results**: Get confidence scores and probability breakdown

## 🚀 Usage

Simply upload an image of a bottle gourd leaf or fruit, and the system will:
- Identify the disease or healthy condition
- Provide confidence percentage
- Show probability breakdown for all classes
- Display model architecture information

## 🔧 Technical Details

- **Framework**: PyTorch with timm library
- **Input Size**: 224x224 pixels
- **Preprocessing**: Standard ImageNet normalization
- **Inference**: Ensemble averaging of model predictions
- **Device**: Supports both CPU and GPU inference

## 📝 Citation

If you use this system in your research, please cite:
```
Bottle Gourd Disease Detection System
Ensemble CNN Models for Plant Disease Classification
```

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting issues
- Suggesting improvements
- Adding new disease classes
- Enhancing model performance

---

**Note**: This system is designed for research and educational purposes. For commercial agricultural use, please validate results with agricultural experts.
