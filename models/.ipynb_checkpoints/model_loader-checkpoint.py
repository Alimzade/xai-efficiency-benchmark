import torch
from torchvision import models, transforms
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
    return model

def preprocess_image(pil_image, model_name='resnet50'):
    """
    Preprocess an input PIL image according to the model's requirements.
    
    For the Vision Transformer, use the dedicated transforms from its weights.
    For other models, use a standard ImageNet preprocessing pipeline.
    
    Parameters:
    - pil_image: A PIL.Image object.
    - model_name: Name of the model to preprocess for.
    
    Returns:
    - A preprocessed tensor ready for inference.
    """
    if model_name == 'vit-b-16':
        transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
