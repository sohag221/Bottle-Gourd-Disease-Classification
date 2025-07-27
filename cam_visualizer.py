"""
CAM Visualization Module for DenseNet Model
Supports: Grad-CAM, Grad-CAM++, EigenCAM, and ScoreCAM
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import io
import base64

class DenseNetCAMVisualizer:
    """
    CAM visualization specifically designed for DenseNet model
    """
    
    def __init__(self, model, device):
        """
        Initialize the CAM visualizer
        
        Args:
            model: DenseNet model
            device: torch.device
        """
        self.model = model
        self.device = device
        
        # Ensure model is in correct dtype and eval mode
        self.model = self.model.float()  # Ensure float32
        self.model.eval()
        
        # Define target layers for DenseNet121
        try:
            # Try to access the target layer
            self.target_layers = [model.features.denseblock4.denselayer16.conv2]
        except AttributeError:
            # Fallback for different DenseNet architectures
            try:
                self.target_layers = [model.features[-1]]  # Last feature layer
            except:
                raise ValueError("Could not identify target layer for DenseNet model")
        
        # Initialize CAM methods with error handling
        self.cam_methods = {}
        
        try:
            self.cam_methods['gradcam'] = GradCAM(model=model, target_layers=self.target_layers)
            print("✅ Grad-CAM initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize Grad-CAM: {e}")
            
        try:
            self.cam_methods['gradcam++'] = GradCAMPlusPlus(model=model, target_layers=self.target_layers)
            print("✅ Grad-CAM++ initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize Grad-CAM++: {e}")
            
        try:
            self.cam_methods['eigencam'] = EigenCAM(model=model, target_layers=self.target_layers)
            print("✅ EigenCAM initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize EigenCAM: {e}")
            
        try:
            self.cam_methods['scorecam'] = ScoreCAM(model=model, target_layers=self.target_layers)
            print("✅ ScoreCAM initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize ScoreCAM: {e}")
        
        if not self.cam_methods:
            raise ValueError("No CAM methods could be initialized")
    
    def preprocess_image_for_cam(self, pil_image):
        """
        Preprocess PIL image for CAM visualization
        
        Args:
            pil_image: PIL Image
            
        Returns:
            tuple: (input_tensor, rgb_img)
        """
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize image
        pil_image = pil_image.resize((224, 224))
        
        # Convert to numpy array and normalize to [0, 1]
        rgb_img = np.array(pil_image) / 255.0
        rgb_img = rgb_img.astype(np.float32)  # Ensure float32 type
        
        # Preprocess for model input
        input_tensor = preprocess_image(rgb_img, 
                                      mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
        
        # Ensure tensor is float32 and on correct device
        input_tensor = input_tensor.float().to(self.device)
        
        return input_tensor, rgb_img
    
    def generate_cam_visualization(self, image, target_class=None, cam_method='gradcam'):
        """
        Generate CAM visualization for a specific method
        
        Args:
            image: PIL Image
            target_class: Target class index (if None, uses predicted class)
            cam_method: CAM method ('gradcam', 'gradcam++', 'eigencam', 'scorecam')
            
        Returns:
            dict: Contains visualization results
        """
        try:
            # Preprocess image
            input_tensor, rgb_img = self.preprocess_image_for_cam(image)
            
            # Ensure model is in eval mode and uses correct dtype
            self.model.eval()
            
            # Get prediction if target_class is not specified
            if target_class is None:
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    target_class = outputs.argmax(dim=1).item()
            
            # Create target for CAM
            targets = [ClassifierOutputTarget(target_class)]
            
            # Get CAM method
            cam_generator = self.cam_methods[cam_method]
            
            # Generate CAM with proper error handling
            with torch.no_grad():
                grayscale_cam = cam_generator(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]  # Take first image from batch
            
            # Ensure grayscale_cam is float32 and normalized
            grayscale_cam = grayscale_cam.astype(np.float32)
            grayscale_cam = np.clip(grayscale_cam, 0, 1)
            
            # Create visualization
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # Convert to PIL Image
            cam_pil = Image.fromarray(cam_image)
            
            # Create heatmap
            heatmap = cm.jet(grayscale_cam)[:, :, :3]  # Remove alpha channel
            heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
            
            return {
                'success': True,
                'original_image': image,
                'cam_overlay': cam_pil,
                'heatmap': heatmap_pil,
                'target_class': target_class,
                'cam_method': cam_method,
                'confidence_score': float(grayscale_cam.max())
            }
            
        except Exception as e:
            print(f"❌ CAM visualization error for {cam_method}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'cam_method': cam_method
            }
    
    def generate_all_cam_visualizations(self, image, target_class=None):
        """
        Generate all CAM visualization methods
        
        Args:
            image: PIL Image
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            dict: Contains all visualization results
        """
        results = {}
        
        # Get prediction if target_class is not specified
        if target_class is None:
            try:
                # Use the main app's preprocessing function to ensure consistency
                from torchvision import transforms
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Use the same transform as the main app
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(image).unsqueeze(0).float().to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    target_class = outputs.argmax(dim=1).item()
                    confidence = probabilities[0, target_class].item()
                    
            except Exception as e:
                print(f"❌ Error getting prediction: {e}")
                target_class = 0  # Default to first class
                confidence = None
        else:
            confidence = None
        
        # Generate visualizations for each available method
        for method_name in self.cam_methods.keys():
            print(f"🔍 Generating {method_name.upper()} visualization...")
            result = self.generate_cam_visualization(image, target_class, method_name)
            results[method_name] = result
            
            if not result['success']:
                print(f"❌ Failed to generate {method_name}: {result['error']}")
        
        # Add metadata
        results['metadata'] = {
            'target_class': target_class,
            'confidence': confidence,
            'image_size': image.size,
            'model_name': 'DenseNet121',
            'available_methods': list(self.cam_methods.keys())
        }
        
        return results
    
    def create_cam_comparison_grid(self, cam_results):
        """
        Create a comparison grid of all CAM methods
        
        Args:
            cam_results: Results from generate_all_cam_visualizations
            
        Returns:
            PIL Image: Comparison grid
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('CAM Visualization Comparison - DenseNet121', fontsize=16, fontweight='bold')
            
            # Original image
            axes[0, 0].imshow(cam_results['metadata']['image_size'])
            axes[0, 0].set_title('Original Image', fontweight='bold')
            axes[0, 0].axis('off')
            
            # CAM methods
            methods = ['gradcam', 'gradcam++', 'eigencam', 'scorecam']
            positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
            
            for method, (row, col) in zip(methods, positions):
                if cam_results[method]['success']:
                    axes[row, col].imshow(cam_results[method]['cam_overlay'])
                    axes[row, col].set_title(f'{method.upper()}', fontweight='bold')
                else:
                    axes[row, col].text(0.5, 0.5, f'Error: {cam_results[method]["error"]}', 
                                      ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].set_title(f'{method.upper()} (Failed)', fontweight='bold', color='red')
                axes[row, col].axis('off')
            
            # Add metadata
            metadata_text = f"""
            Predicted Class: {cam_results['metadata']['target_class']}
            Confidence: {cam_results['metadata']['confidence']:.4f}
            Model: {cam_results['metadata']['model_name']}
            """
            axes[1, 2].text(0.1, 0.5, metadata_text, transform=axes[1, 2].transAxes, 
                           fontsize=12, verticalalignment='center')
            axes[1, 2].set_title('Prediction Info', fontweight='bold')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Convert to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            comparison_image = Image.open(buf)
            plt.close()
            
            return comparison_image
            
        except Exception as e:
            print(f"❌ Error creating comparison grid: {e}")
            return None
    
    def pil_to_base64(self, pil_image):
        """
        Convert PIL Image to base64 string for web display
        
        Args:
            pil_image: PIL Image
            
        Returns:
            str: base64 encoded image
        """
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def generate_web_cam_results(self, image, target_class=None):
        """
        Generate CAM results formatted for web display
        
        Args:
            image: PIL Image
            target_class: Target class index
            
        Returns:
            dict: Web-formatted results
        """
        # Generate all CAM visualizations
        cam_results = self.generate_all_cam_visualizations(image, target_class)
        
        # Format for web
        web_results = {
            'success': True,
            'metadata': cam_results['metadata'],
            'visualizations': {}
        }
        
        # Convert each method's results to base64
        for method_name in ['gradcam', 'gradcam++', 'eigencam', 'scorecam']:
            if cam_results[method_name]['success']:
                web_results['visualizations'][method_name] = {
                    'success': True,
                    'cam_overlay': self.pil_to_base64(cam_results[method_name]['cam_overlay']),
                    'heatmap': self.pil_to_base64(cam_results[method_name]['heatmap']),
                    'confidence_score': cam_results[method_name]['confidence_score']
                }
            else:
                web_results['visualizations'][method_name] = {
                    'success': False,
                    'error': cam_results[method_name]['error']
                }
        
        # Create comparison grid
        comparison_grid = self.create_cam_comparison_grid(cam_results)
        if comparison_grid:
            web_results['comparison_grid'] = self.pil_to_base64(comparison_grid)
        
        return web_results

def create_densenet_cam_visualizer(model, device):
    """
    Factory function to create DenseNet CAM visualizer
    
    Args:
        model: DenseNet model
        device: torch.device
        
    Returns:
        DenseNetCAMVisualizer instance
    """
    return DenseNetCAMVisualizer(model, device)
