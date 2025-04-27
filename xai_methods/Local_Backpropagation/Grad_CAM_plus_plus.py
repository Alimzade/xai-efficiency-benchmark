import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit
from torchvision import models
from models.label_utils import get_label_mapping
from utils.data_utils import process_data_tuple

def generate_attribution(image, predicted_class, model, target_layer=None):
    """
    Generate Grad-CAM++ attribution for the given image and class.
    
    Parameters:
    - image: Tensor representing the input image.
    - predicted_class: Predicted class label for which to generate the attribution.
    - model: Pretrained model.
    - target_layer: Optional specific layer to use for Grad-CAM++.
    
    Returns:
    - attribution: Grad-CAM++ heatmap.
    """
    if target_layer is None:
        if isinstance(model, models.ResNet):
            target_layer = model.layer4[-1]
        elif isinstance(model, models.DenseNet):
            target_layer = model.features[-1]
        elif hasattr(model, 'features'):
            target_layer = model.features[-1]
        elif isinstance(model, models.VisionTransformer):
            # Use the last encoder layer's LayerNorm (ln_1) as target.
            target_layer = model.encoder.layers[-1].ln_1
        elif isinstance(model, models.SwinTransformer):
            target_layer = model.features[-1][-1].mlp[0]
        elif isinstance(model, models.RegNet):
            target_layer = model.trunk_output
        elif isinstance(model, models.MobileNetV3):
            target_layer = model.features[-1]
        else:
            raise ValueError(f"Unsupported model architecture: {type(model).__name__}")
    
    activations = []
    gradients = []
    
    def forward_hook(module, inp, output):
        activations.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)
    
    output = model(image)
    score = output[0, predicted_class.item()]
    
    model.zero_grad()
    score.backward(retain_graph=True)
    
    act = activations[0]
    grad = gradients[0]
    
    # If model is VisionTransformer, reshape token outputs
    if isinstance(model, models.VisionTransformer):
        # Expect shape [1, 197, D]; remove the first token (CLS)
        act = act[:, 1:, :]  # [1, 196, D]
        grad = grad[:, 1:, :]
        B, N, D = act.shape
        H = W = int(np.sqrt(N))
        if H * W != N:
            forward_handle.remove()
            backward_handle.remove()
            raise ValueError(f"Cannot reshape {N} tokens into a square feature map.")
        act = act.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [1, D, H, W]
        grad = grad.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [1, D, H, W]
    
    activations_val = act
    gradients_val = grad
    
    grads_squared = gradients_val ** 2
    grads_cubed = gradients_val ** 3
    
    numerator = grads_squared
    denominator = 2 * grads_squared + (activations_val * grads_cubed).sum(dim=(2,3), keepdim=True)
    eps = 1e-8
    alphas = numerator / (denominator + eps)
    
    weights = (alphas * F.relu(gradients_val)).sum(dim=(2,3), keepdim=True)
    
    cam = (weights * activations_val).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    
    cam = F.interpolate(cam, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
    
    cam_min = cam.min()
    cam_max = cam.max()
    attribution = (cam - cam_min) / (cam_max - cam_min + eps)
    
    forward_handle.remove()
    backward_handle.remove()
    
    return attribution

def warm_up(model):
    dummy_image = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device, requires_grad=True)
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
