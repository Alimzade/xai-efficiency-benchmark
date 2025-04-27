import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit
from utils.data_utils import process_data_tuple
from models.label_utils import get_label_mapping
from torchvision import models

def get_classifier_weights(model):
    """
    Retrieve classifier weights from the model by checking common attributes.
    """
    # Check if model has a direct 'fc' attribute (e.g., ResNet)
    if hasattr(model, 'fc'):
        return model.fc.weight
    # Check if model has a 'classifier' attribute
    elif hasattr(model, 'classifier'):
        classifier = model.classifier
        # If classifier is a Sequential container, use the last module
        if isinstance(classifier, torch.nn.Sequential):
            if isinstance(classifier[-1], torch.nn.Linear):
                return classifier[-1].weight
            else:
                raise ValueError("Classifier is sequential but does not end with a Linear layer.")
        elif isinstance(classifier, torch.nn.Linear):
            return classifier.weight
    # Check for 'head' attribute (common in transformer-based models like Swin)
    elif hasattr(model, 'head'):
        head = model.head
        if isinstance(head, torch.nn.Linear):
            return head.weight
    raise ValueError("Model does not have a recognizable classifier layer for CAM.")

def generate_attribution(image, predicted_class, model, target_layer=None):
    """
    Unified CAM attribution for CNN models (ResNet, DenseNet, etc.) and VisionTransformer models.
    
    Parameters:
    - image: input tensor [1, 3, H, W]
    - predicted_class: int or tensor indicating predicted class
    - model: pretrained torchvision model
    - target_layer: optional; will be auto-selected if None
    
    Returns:
    - CAM heatmap tensor [1, 1, H, W]
    """
    activations = []

    def forward_hook(module, inp, output):
        activations.append(output)

    # 1. Identify and hook appropriate target_layer:
    if target_layer is None:
        if isinstance(model, models.ResNet):
            target_layer = model.layer4[-1]
        elif isinstance(model, models.DenseNet):
            target_layer = model.features[-1]
        elif hasattr(model, 'features'):
            target_layer = model.features[-1]
        elif isinstance(model, models.VisionTransformer):
            target_layer = model.encoder.layers[-1].ln_1
        else:
            raise ValueError(f"Unsupported model: {type(model).__name__} for CAM")

    handle = target_layer.register_forward_hook(forward_hook)
    model_output = model(image)
    handle.remove()

    if len(activations) == 0:
        raise RuntimeError("Hook failed; no activations captured.")

    activation_maps = activations[0]

    # 2. Handle ViT-specific processing:
    if isinstance(model, models.VisionTransformer):
        # activation_maps shape: [1, 197, 768]; remove class token
        activation_maps = activation_maps[:, 1:, :]  # [1,196,768]
        B, N, D = activation_maps.shape
        H = W = int(np.sqrt(N))
        activation_maps = activation_maps.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [1,768,H,W]

        # Classifier weights for ViT:
        classifier_weights = model.heads[0].weight  # [num_classes, 768]
    else:
        # CNNs (ResNet, DenseNet, etc.): directly [1,C,H,W]
        classifier_weights = get_classifier_weights(model)  # [num_classes, C]

    # 3. Compute CAM using classifier weights for predicted class:
    if isinstance(predicted_class, torch.Tensor):
        predicted_class = predicted_class.item()

    class_weights = classifier_weights[predicted_class]  # [C]

    # Weighted sum across channels:
    cam = torch.sum(class_weights.view(1, -1, 1, 1) * activation_maps, dim=1, keepdim=True)  # [1,1,H,W]

    # 4. Relu, upscale, and normalize to [0,1]:
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=image.shape[2:], mode='bilinear', align_corners=False)

    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    return cam


def warm_up(model):
    dummy_image = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
    with torch.no_grad():
        output = model(dummy_image)
        _, predicted_class = torch.max(output, 1)
    _ = generate_attribution(dummy_image, predicted_class, model)

def visualize_attribution(image, attribution, label, label_names, model, model_name, save_path=None):
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    
    predicted_label, true_label = get_label_mapping(
        model_name=model_name,
        predicted_class=predicted_class,
        label=label,
        label_names=label_names,
    )
    
    print(f"Model: {model_name}, Predicted Class: {predicted_label}, True Class: {true_label}")
    
    heatmap = attribution.squeeze().cpu().detach().numpy()
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)
    
    heatmap_resized = cv2.resize(heatmap_colored, (img_np.shape[1], img_np.shape[0]))
    overlayed_img = cv2.addWeighted(img_np, 0.6, heatmap_resized, 0.4, 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_np)
    axes[0].set_title('Input' if true_label == "Unknown" else f'True: {true_label}')
    axes[0].axis('off')
    axes[1].imshow(overlayed_img)
    axes[1].set_title(f'Predicted: {predicted_label}')
    axes[1].axis('off')
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        visualize_attribution.save_count += 1
        print(f"Saved visualization at {save_path}")
    
    plt.show()
    plt.close(fig)

visualize_attribution.save_count = 0

def measure_avg_time_across_images(data_source, model, generate_attribution):
    times = []
    for data_item in data_source:
        image, label = process_data_tuple(data_item, model)
        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)
        start_time = timeit.default_timer()
        _ = generate_attribution(image, predicted_class, model)
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
        torch.cuda.empty_cache()
    avg_time_taken = sum(times) / len(times)
    return avg_time_taken, times


def warm_up(model):
    dummy_image = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
    with torch.no_grad():
        output = model(dummy_image)
        _, predicted_class = torch.max(output, 1)
    _ = generate_attribution(dummy_image, predicted_class, model)

def visualize_attribution(image, attribution, label, label_names, model, model_name, save_path=None):
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    predicted_label, true_label = get_label_mapping(
        model_name=model_name,
        predicted_class=predicted_class,
        label=label,
        label_names=label_names,
    )
    print(f"Model: {model_name}, Predicted Class: {predicted_label}, True Class: {true_label}")
    heatmap = attribution.squeeze().cpu().detach().numpy()
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap_colored, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    overlayed_img = cv2.addWeighted(img_np, 0.6, heatmap_resized, 0.4, 0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.5)
    axes[0].imshow(img_np)
    axes[0].set_title('Input' if true_label == "Unknown" else f'True: {true_label}')
    axes[0].axis('off')
    axes[1].imshow(overlayed_img)
    axes[1].set_title(f'Predicted: {predicted_label}')
    axes[1].axis('off')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        visualize_attribution.save_count += 1
        print(f"Saved visualization at {save_path}")
    plt.show()
    plt.close(fig)

visualize_attribution.save_count = 0

def measure_avg_time_across_images(data_source, model, generate_attribution):
    times = []
    for data_item in data_source:
        image, label = process_data_tuple(data_item, model)
        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)
        start_time = timeit.default_timer()
        _ = generate_attribution(image, predicted_class, model)
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
        torch.cuda.empty_cache()
    avg_time_taken = sum(times) / len(times)
    return avg_time_taken, times

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'vit-b-16'
    from torchvision.models import ViT_B_16_Weights
    model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    print(f"Loaded {model_name} successfully on {device}!")
