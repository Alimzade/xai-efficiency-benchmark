import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.models import (
    ResNet50_Weights,
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    Swin_T_Weights,
    RegNet_Y_8GF_Weights,
    MobileNet_V3_Large_Weights,
    DenseNet121_Weights,
    ViT_B_16_Weights
)

# Updated MODEL_ZOO dictionary
MODEL_ZOO = {
    'resnet50': (models.resnet50, ResNet50_Weights.DEFAULT),
    'convnext-t': (models.convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT),  # Tiny variant
    'efficientnet-b0': (models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
    'swin-t': (models.swin_t, Swin_T_Weights.IMAGENET1K_V1),  # Swin Transformer Tiny
    'regnet-y-8gf': (models.regnet_y_8gf, RegNet_Y_8GF_Weights.IMAGENET1K_V1),  # RegNetY
    'mobilenet-v3-large': (models.mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V1),  # MobileNetV3
    'densenet121': (models.densenet121, DenseNet121_Weights.IMAGENET1K_V1),  # DenseNet
    'vit-b-16': (models.vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1)  # Vision Transformer
}

# Patch forward methods and replace relu modules for DeepLift compatibility
def _patched_bottleneck_forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu3(out)

    return out

def _patched_basicblock_forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
        identity = self.downsample(x)

    out += identity
    out = self.relu2(out)

    return out

# Set the classes' forward methods to the patched versions globally:
Bottleneck.forward = _patched_bottleneck_forward
BasicBlock.forward = _patched_basicblock_forward

def load_model(model_name='resnet50', device=None):
    """
    Load a pretrained model based on the model name, using updated weights parameter.
    
    Parameters:
    - model_name: Name of the model to load (default is ResNet-50).
    - device: Device to load the model onto (e.g., 'cuda' or 'cpu').
    
    Returns:
    - Pretrained model set to evaluation mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if model_name is in MODEL_ZOO
    if model_name in MODEL_ZOO:
        # Torchvision models
        model_fn, weights = MODEL_ZOO[model_name]
        model = model_fn(weights=weights).to(device)
    else:
        available_models = ", ".join(MODEL_ZOO.keys())
        raise ValueError(f"Model '{model_name}' is not implemented. Available models: {available_models}")
    
    model.eval()
    
    # For ResNet architectures, Bottleneck and BasicBlock reuse self.relu module
    # multiple times, which crashes Captum's DeepLift. We replace them with separate instances.
    for module in model.modules():
        if isinstance(module, Bottleneck):
            module.relu1 = torch.nn.ReLU(inplace=False)
            module.relu2 = torch.nn.ReLU(inplace=False)
            module.relu3 = torch.nn.ReLU(inplace=False)
        elif isinstance(module, BasicBlock):
            module.relu1 = torch.nn.ReLU(inplace=False)
            module.relu2 = torch.nn.ReLU(inplace=False)
            
    # DeepLift/DeepLiftShap require out-of-place activations (e.g. ReLU(inplace=False))
    for module in model.modules():
        if hasattr(module, 'inplace'):
            module.inplace = False
            
    return model

# Models that strictly require a fixed size (224x224) due to positional embeddings or window constraints
FIXED_SIZE_MODELS = ['vit-b-16', 'swin-t']

def preprocess_image(pil_image, model_name='resnet50', target_size=224):
    """
    Preprocess an input PIL image according to the model's requirements.
    
    Parameters:
    - pil_image: A PIL.Image object.
    - model_name: Name of the model to preprocess for.
    - target_size: The desired square dimension for input (default 224).
    
    Returns:
    - A preprocessed tensor ready for inference.
    """
    if model_name in FIXED_SIZE_MODELS:
        # Use the standard weights-defined transforms for fixed-size models
        if model_name == 'vit-b-16':
            transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
        elif model_name == 'swin-t':
            transform = Swin_T_Weights.IMAGENET1K_V1.transforms()
    else:
        # Calculate resize value relative to target_size (standard is 256/224 ratio ≈ 1.14)
        resize_val = int(target_size * (256 / 224))
        transform = transforms.Compose([
            transforms.Resize(resize_val),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transform(pil_image)

# Example Usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'vit-b-16'  # Now supported: Vision Transformer
    model = load_model(model_name=model_name, device=device)
    print(f"Loaded {model_name} successfully on {device}!")
