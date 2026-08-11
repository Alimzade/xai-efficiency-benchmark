"""
Core Globals Module
Contains globally shared instances like the SessionManager and constant paths.
"""
import os
import logging
import torch

from backend.session_manager import SessionManager

# Setup basic module logger
logger = logging.getLogger(__name__)

# Core Global Paths and Singletons
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sm = SessionManager()

from utils.model_loader import MODEL_ZOO, FIXED_SIZE_MODELS

# Default Configurations
model_opts = list(MODEL_ZOO.keys())
fixed_size_models = FIXED_SIZE_MODELS
min_input_size = 32
default_input_size = 224
region_based_methods = ["Occlusion", "LIME"]
pixel_based_methods = ["Saliency", "Integrated_Gradients", "Guided_Backprop", "Input_X_Gradient", "Gradient_Shap", "DeepLift", "DeepLift_Shap", "Grad_CAM"]
xai_opts = ["Saliency", "Integrated_Gradients", "Guided_Backprop", "Input_X_Gradient", "Gradient_Shap", "DeepLift", "DeepLift_Shap", "Grad_CAM", "Occlusion", "LIME"]

# Hardware Device Detection
has_cuda = torch.cuda.is_available()
has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
default_device_mode = "GPU (CUDA)" if has_cuda else "GPU (MPS)" if has_mps else "CPU"

# Centralized XAI Parameter Configurations
method_configs = {
    "Integrated_Gradients": {
        "n_steps": {
            "type": "int", "default": "50", "label": "Steps", "choices": ["25", "50", "100", "250", "Custom"],
            "help": "Number of steps along the path from baseline to input.\n\n- Default: 50\n- Higher is more mathematically accurate but slower."
        },
        "internal_batch_size": {
            "type": "int", "default": "2", "label": "Internal Batch Size", "choices": ["1", "2", "4", "8", "Custom"],
            "help": "Mini-batch size to chunk the steps.\n\n- Default: 2\n- Set to 2-4 to prevent Out-Of-Memory (OOM) on large models."
        },
        "baseline_mode": {
            "type": "select", "default": "Zeros (Black)", "label": "Baseline Mode", "choices": ["Zeros (Black)", "Ones (White)", "Input Mean"],
            "help": "Baseline reference image for attribution.\n\n- Default: Zeros (Black)"
        }
    },
    "Gradient_Shap": {
        "n_samples": {
            "type": "int", "default": "10", "label": "Samples", "choices": ["5", "10", "20", "50", "Custom"],
            "help": "Number of baseline samples to average.\n\n- Default: 10\n- Suggest 10-20 for stable results."
        },
        "stdevs": {
            "type": "float", "default": "0.0001", "label": "Stdevs (Noise)", "choices": ["0.0001", "0.01", "0.1", "1.0", "Custom"],
            "help": "Standard deviation of Gaussian noise added to inputs.\n\n- Default: 0.0001"
        },
        "baseline_mode": {
            "type": "select", "default": "Zeros & Mean", "label": "Baseline Mode", "choices": ["Zeros & Mean", "Zeros Only", "Ones Only"],
            "help": "Distribution of baselines to sample from.\n\n- Default: Zeros & Mean"
        }
    },
    "Occlusion": {
        "sliding_window_shapes": {
            "type": "int", "default": "15", "label": "Patch Size (px)", "choices": ["8", "15", "30", "50", "Custom"],
            "help": "Size of the square occlusion patch.\n\n- Default: 15\n- Larger is faster but coarser."
        },
        "strides": {
            "type": "int", "default": "8", "label": "Stride (px)", "choices": ["4", "8", "15", "30", "Custom"],
            "help": "Step size of the sliding window.\n\n- Default: 8\n- Smaller is more detailed but much slower."
        },
        "occlude_color": {
            "type": "int", "default": "0", "label": "Occlude Color", "choices": ["0", "127", "255", "Custom"],
            "help": "Pixel value to fill the occlusion patch.\n\n- Default: 0 (Black)\n- Can be 0, 1 (White), or mean."
        }
    },
    "LIME": {
        "n_samples": {
            "type": "int", "default": "500", "label": "Perturbation Samples", "choices": ["100", "500", "1000", "5000", "Custom"],
            "help": "Number of samples to perturb and fit the surrogate model on.\n\n- Default: 500\n- Warning: High values (>1000) are very slow."
        },
        "perturbations_per_eval": {
            "type": "int", "default": "10", "label": "Batch Size", "choices": ["1", "10", "32", "64", "Custom"],
            "help": "Number of perturbations evaluated simultaneously.\n\n- Default: 10\n- Higher values use more memory but run faster."
        },
        "n_segments": {
            "type": "int", "default": "50", "label": "Superpixels (Segments)", "choices": ["10", "50", "100", "200", "Custom"],
            "help": "Number of superpixels to divide the image into using SLIC.\n\n- Default: 50"
        }
    }
}
