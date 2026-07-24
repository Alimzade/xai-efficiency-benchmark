"""
Core Globals Module
Contains globally shared instances like the SessionManager and constant paths.
"""
import os
import logging
import torch

from gui.backend.session_manager import SessionManager

# Setup basic module logger
logger = logging.getLogger(__name__)

# Core Global Paths and Singletons
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sm = SessionManager()

# Default Configurations
model_opts = ['resnet50', 'convnext-t', 'efficientnet-b0', 'swin-t', 'regnet-y-8gf', 'mobilenet-v3-large', 'densenet121', 'vit-b-16']
xai_opts = ["Saliency", "Integrated_Gradients", "Guided_Backprop", "Input_X_Gradient", "Gradient_Shap", "DeepLift", "DeepLift_Shap", "Grad_CAM", "Occlusion", "Lime"]

# Hardware Device Detection
has_cuda = torch.cuda.is_available()
has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
default_device_mode = "GPU (CUDA)" if has_cuda else "GPU (MPS)" if has_mps else "CPU"
