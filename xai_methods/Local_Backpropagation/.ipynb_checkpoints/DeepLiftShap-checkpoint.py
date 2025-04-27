import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import DeepLiftShap
from utils.data_utils import process_data_tuple
from models.label_utils import get_label_mapping
import timeit


class CustomReLU(nn.Module):
    def __init__(self, inplace=False):
        super(CustomReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)


def replace_relu_with_custom(module):
    """
    Replace nn.ReLU layers with CustomReLU layers in the model.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, CustomReLU(inplace=child.inplace))
        else:
            replace_relu_with_custom(child)


def generate_attribution(image, predicted_class, model, baseline=None):
    """
    Generate DeepLIFT SHAP attribution for the given image and class.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    replace_relu_with_custom(model)  # Replace ReLU with CustomReLU

    # Initialize DeepLiftShap
    shap = DeepLiftShap(model)

    # Default baseline distribution if none provided
    if baseline is None:
        baseline = torch.zeros_like(image).repeat(5, 1, 1, 1).to(device)

    # Generate attributions using Deep SHAP
    attribution = shap.attribute(image, baselines=baseline, target=predicted_class.item())
    return attribution


def warm_up(model):
    """
    Run a warm-up pass to ensure memory and computation stability.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_image = torch.zeros(1, 3, 224, 224, device=device, requires_grad=True)

    baseline_distribution = dummy_image.repeat(5, 1, 1, 1)

    with torch.no_grad():
        output = model(dummy_image)
        _, predicted_class = torch.max(output, 1)

    _ = generate_attribution(dummy_image, predicted_class, model, baseline=baseline_distribution)


def visualize_attribution(image, attribution, label, label_names, model, model_name, save_path=None):
    """
    Visualize the DeepLIFT SHAP attribution overlayed on the original image, including predicted class.

    Parameters:
    - image: Original input image tensor.
    - attribution: DeepLIFT SHAP attribution.
    - label: True label of the image (or None for URL data).
    - label_names: List of class names specific to the dataset (or None for URL data).
    - model: Pretrained model used for predictions.
    - model_name: The name of the model (e.g., 'resnet50', 'convnext-t', 'efficientnet-b0').
    - save_path: Optional file path to save the figure directly.

    Returns:
    - None
    """

    # Perform prediction
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)

    # Get label mappings
    predicted_label, true_label = get_label_mapping(
        model_name=model_name,
        predicted_class=predicted_class,
        label=label,
        label_names=label_names,
    )

    print(f"Model: {model_name}, Predicted Class: {predicted_label}, True Class: {true_label}")

    # Convert and normalize attribution
    attribution = attribution.squeeze().cpu().detach().numpy()
    if attribution.ndim == 3:
        attribution = np.sum(np.abs(attribution), axis=0)
    epsilon = 1e-8
    attribution = (attribution - attribution.min()) / (attribution.max() + epsilon)
    gamma = 0.5
    attribution = np.power(attribution, gamma)

    # Prepare attribution for colormap application
    attribution = (attribution * 255).astype(np.uint8)
    attribution_colored = cv2.applyColorMap(attribution, cv2.COLORMAP_JET)

    # Process original image for visualization
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)
    overlayed_img = cv2.addWeighted(img_np, 0.5, attribution_colored, 0.5, 0)

    # Plot images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Increased size for better spacing
    fig.subplots_adjust(wspace=0.5)  # Add spacing between subplots

    # Plot original image
    axes[0].imshow(img_np)
    if true_label == "Unknown":
        axes[0].set_title(f'Input')
    else:
        axes[0].set_title(f'True: {true_label}')
    axes[0].axis('off')

    # Plot heatmap overlay
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


# Initialize the save counter
visualize_attribution.save_count = 0


def measure_avg_time_across_images(data_source, model, generate_attribution):
    """
    Measure average time taken to generate attributions across images.
    
    Parameters:
    - data_source: Data source for images, either a DataLoader or list of tuples (image, label).
    - model: Pretrained model.
    - generate_attribution: Function to generate attributions.
    
    Returns:
    - Average time taken and list of times per image.
    """
    times = []

    # Process each data item
    for data_tuple in data_source:
        # Use the provided utility function to process the data tuple
        image, label = process_data_tuple(data_tuple, model)
        
        # Predict class
        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)

        # Measure time for attribution generation
        start_time = timeit.default_timer()
        _ = generate_attribution(image, predicted_class, model)
        end_time = timeit.default_timer()

        times.append(end_time - start_time)
        torch.cuda.empty_cache()

    # Calculate average time
    avg_time_taken = sum(times) / len(times)
    return avg_time_taken, times
