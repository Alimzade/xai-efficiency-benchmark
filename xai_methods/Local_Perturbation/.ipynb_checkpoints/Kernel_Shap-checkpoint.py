import torch
import shap
import numpy as np
import cv2
import matplotlib.pyplot as plt
import timeit
from utils.data_utils import process_data_tuple
from models.label_utils import get_label_mapping

def generate_attribution(image, predicted_class, model, nsamples=100):
    """
    Generate Kernel SHAP attributions for the given image and class using flattened input.
    This method uses SHAP's KernelExplainer to compute approximate Shapley values.
    It is completely model-agnostic and works by perturbing the flattened input.
    
    Parameters:
    - image: Input tensor of shape [1, 3, H, W].
    - predicted_class: The target class for which to explain the prediction.
    - model: Pretrained model.
    - nsamples: Number of samples for the KernelExplainer.
    
    Returns:
    - attribution: A 2D attribution map (heatmap) with values normalized to [0,1].
    """
    device = image.device
    B, C, H, W = image.shape
    num_features = C * H * W

    # Flatten image: shape becomes [1, num_features]
    img_np = image.cpu().numpy().reshape((B, num_features))
    # Use a black image as background: shape [1, num_features]
    background = np.zeros((1, num_features))
    
    def predict_fn(x):
        # x is a numpy array of shape [batch, num_features]
        batch = x.shape[0]
        x_reshaped = x.reshape((batch, C, H, W))
        x_tensor = torch.tensor(x_reshaped, dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(x_tensor)
            probabilities = torch.nn.functional.softmax(out, dim=1)
        return probabilities.cpu().numpy()
    
    explainer = shap.KernelExplainer(predict_fn, background, l1_reg=0.001)

    shap_values = explainer.shap_values(img_np, nsamples=nsamples)
    
    target_class = predicted_class.item() if isinstance(predicted_class, torch.Tensor) else predicted_class
    # shap_values is a list of arrays, one per class, each with shape [1, num_features]
    attr_flat = shap_values[target_class]  # shape: [1, num_features]
    # Reshape to [3, H, W]
    attr_reshaped = attr_flat.reshape((C, H, W))
    # Average over channels to get a single heatmap: shape: [H, W]
    attr_mean = np.mean(attr_reshaped, axis=0)
    
    # Normalize to [0, 1]
    attr_min, attr_max = attr_mean.min(), attr_mean.max()
    if attr_max - attr_min > 1e-8:
        attr_norm = (attr_mean - attr_min) / (attr_max - attr_min)
    else:
        attr_norm = np.zeros_like(attr_mean)
    
    attribution = torch.tensor(attr_norm, device=device).float()
    return attribution

def warm_up(model):
    """
    Run a warm-up pass for Kernel SHAP.
    """
    dummy_image = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device)
    with torch.no_grad():
        out = model(dummy_image)
        _, predicted_class = torch.max(out, 1)
    _ = generate_attribution(dummy_image, predicted_class, model, nsamples=50)

def visualize_attribution(image, attribution, label, label_names, model, model_name, save_path=None):
    """
    Visualize the Kernel SHAP heatmap overlaid on the original image.
    
    Parameters:
    - image: Original input image tensor.
    - attribution: Attribution heatmap from Kernel SHAP.
    - label: True label (or None).
    - label_names: List of class names.
    - model: Pretrained model.
    - model_name: Model name.
    - save_path: Optional file path for saving the visualization.
    """
    with torch.no_grad():
        out = model(image)
        _, predicted_class = torch.max(out, 1)
    
    predicted_label, true_label = get_label_mapping(
        model_name=model_name,
        predicted_class=predicted_class,
        label=label,
        label_names=label_names
    )
    print(f"Model: {model_name}, Predicted: {predicted_label}, True: {true_label}")
    
    # Convert image to NumPy array for visualization (assume ImageNet normalization)
    img_np = image.squeeze().permute(1,2,0).cpu().numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)
    
    # Prepare the attribution heatmap.
    attr_np = attribution.cpu().numpy()
    attr_np = (attr_np * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(attr_np, cv2.COLORMAP_JET)
    
    overlayed_img = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    axes[0].imshow(img_np)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(overlayed_img)
    axes[1].set_title(f"Predicted: {predicted_label}")
    axes[1].axis("off")
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    
    plt.show()
    plt.close(fig)

def measure_avg_time_across_images(data_source, model, generate_attribution):
    times = []
    for data_item in data_source:
        image, label = process_data_tuple(data_item, model)
        with torch.no_grad():
            out = model(image)
            _, predicted_class = torch.max(out, 1)
        start_time = timeit.default_timer()
        _ = generate_attribution(image, predicted_class, model)
        end_time = timeit.default_timer()
        times.append(end_time - start_time)
    avg_time_taken = sum(times)/len(times)
    return avg_time_taken, times
