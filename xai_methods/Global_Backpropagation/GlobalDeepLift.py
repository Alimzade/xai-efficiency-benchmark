import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import DeepLift
import timeit

# Replace ReLU with CustomReLU (required for DeepLIFT)
class CustomReLU(nn.Module):
    def __init__(self, inplace=False):
        super(CustomReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)

def replace_relu_with_custom(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, CustomReLU(inplace=child.inplace))
        else:
            replace_relu_with_custom(child)

def generate_attribution(images, predicted_classes, model, baseline=None):
    device = images.device
    replace_relu_with_custom(model)
    deeplift = DeepLift(model)

    if baseline is None:
        baseline = torch.zeros_like(images).to(device)

    attributions = []
    for image, target in zip(images, predicted_classes):
        image = image.clone().detach().requires_grad_(True).unsqueeze(0)
        attribution = deeplift.attribute(image, baselines=baseline[:1], target=target.item())
        attributions.append(attribution.cpu().detach().numpy())

    aggregated_attributions = np.sum(attributions, axis=0)
    aggregated_attributions -= aggregated_attributions.min()
    aggregated_attributions /= aggregated_attributions.max() + 1e-8

    return aggregated_attributions


def warm_up(model):
    """
    Run a warm-up pass to ensure memory and computation stability.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_image = torch.zeros(1, 3, 224, 224, device=device, requires_grad=True)

    # Create a baseline matching the dummy image shape
    baseline = torch.zeros_like(dummy_image).to(device)

    with torch.no_grad():
        output = model(dummy_image)
        _, predicted_class = torch.max(output, 1)

    _ = generate_attribution(dummy_image, predicted_class, model, baseline=baseline)


def measure_avg_time_across_images(data_source, model, generate_attribution, url_data=False):
    """
    Measure average time taken to generate attributions across images for global DeepLIFT.

    Parameters:
    - data_source: Data source for images, either a DataLoader or list of images.
    - model: Pretrained model.
    - generate_attribution: Function to generate global DeepLIFT attributions.
    - url_data: Boolean flag for URL data, which has no labels.

    Returns:
    - Average time taken, list of times per image, representative image, and aggregated DeepLIFT attribution.
    """
    times = []
    attributions = []
    representative_image = None

    for idx, data in enumerate(data_source):
        # For URL data, `data` is a single image; otherwise, it's a tuple (image, label)
        image = data if url_data else data[0]
        image = image.to(next(model.parameters()).device)

        # Set the representative image to the first image processed
        if representative_image is None:
            representative_image = image.cpu().numpy().squeeze()

        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)

        # Timing attribution generation
        start_time = timeit.default_timer()
        attribution = generate_attribution(image, predicted_class, model)
        end_time = timeit.default_timer()

        attributions.append(attribution)
        times.append(end_time - start_time)
        torch.cuda.empty_cache()

    avg_time_taken = sum(times) / len(times)
    aggregated_attribution = np.mean(attributions, axis=0)  # Aggregate attributions
    return avg_time_taken, times, representative_image, aggregated_attribution

def visualize_aggregated_attribution(aggregated_attribution, save_path=None):
    # Remove singleton dimensions if present
    aggregated_attribution = np.squeeze(aggregated_attribution)

    # Collapse to 2D if the attribution has 3 channels
    if aggregated_attribution.ndim == 3 and aggregated_attribution.shape[0] == 3:
        aggregated_attribution = np.mean(aggregated_attribution, axis=0)

    # Normalize for better visualization
    epsilon = 1e-8
    aggregated_attribution -= aggregated_attribution.min()
    aggregated_attribution /= aggregated_attribution.max() + epsilon

    # Plot the heatmap
    plt.imshow(aggregated_attribution, cmap='plasma')
    plt.colorbar()
    plt.title("Aggregated Global DeepLIFT Attribution")

    # Save if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        print(f"Global DeepLIFT attribution saved at {save_path}")

    plt.show()
