# Bottle Gourd Disease Classification - GitHub Pages Deployment

This branch is specifically configured for GitHub Pages deployment of the Bottle Gourd Disease Classification interface.

## What Works in this Version

- Full web interface with all UI elements
- Smart demo mode that analyzes image characteristics to provide varied predictions
- All visual features and styling

## How the Demo Mode Works

When hosted on GitHub Pages or other static hosting platforms (like Vercel):

1. The app analyzes your uploaded image's color characteristics
2. It uses these characteristics to make a simulated "prediction"
3. For example:
   - Images with more yellow content may be classified as "Yellow mosaic virus"
   - Images with more red content may be classified as "Anthracnose fruit rot"
   - Very green images may be classified as "Fresh leaf"

**Important**: These are simulated predictions based on basic image analysis, NOT real AI model inference.

## For Actual AI-Powered Predictions

For the full application with actual AI model inference:

1. Clone the main repository
2. Follow the setup instructions in the main README
3. Run the Flask application locally

## Demo Mode Banner

A blue notification banner will appear at the bottom of the screen to clearly indicate when the app is running in demo mode.
