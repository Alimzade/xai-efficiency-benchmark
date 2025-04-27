import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import GradientShap
import timeit
import statistics

# Define the global generate_attribution function
def generate_attribution(images, predicted_classes, model):
    """
    Generate aggregated Gradient SHAP attribution for multiple images.
    
    Parameters:
    - images: Batch of image tensors.
    - predicted_classes: Batch of predicted class labels for each image.
    - model: Pretrained model to be used with Gradient SHAP.
    
    Returns:
    - Aggregated attribution map for all images.
    """
    gradient_shap = GradientShap(model)
    baseline_dist = torch.randn((5,) + images.shape[1:]).to(images.device)  # 5 baselines for each image
    # Aggregate attributions across all images
    attributions = []
    for image, target in zip(images, predicted_classes):
        attribution = gradient_shap.attribute(image.unsqueeze(0), baselines=baseline_dist, target=target.item())
        attributions.append(attribution.cpu().detach().numpy())
    # Aggregate attributions across the batch to get a global view
    return np.mean(attributions, axis=0)  # Average SHAP values for a global perspective

# Warm-up function to stabilize GPU memory usage if needed
def warm_up(model):
    dummy_image = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device, requires_grad=True)
    with torch.no_grad():
        output = model(dummy_image)
        _, predicted_class = torch.max(output, 1)
    _ = generate_attribution(dummy_image, predicted_class, model)

# Visualize the aggregated global attribution as a heatmap
def visualize_aggregated_attribution(aggregated_attribution, save_path=None):
    # Remove singleton dimensions
    aggregated_attribution = np.squeeze(aggregated_attribution)
    #print("Shape after squeezing:", aggregated_attribution.shape)  # Debugging line

    # If it still has three channels, average across them to reduce to 2D
    if aggregated_attribution.ndim == 3 and aggregated_attribution.shape[0] == 3:
        aggregated_attribution = aggregated_attribution.mean(axis=0)  # Collapse to (224, 224)
    #print("Shape before plotting:", aggregated_attribution.shape)  # Debugging line

    # Plot the 2D heatmap
    plt.imshow(aggregated_attribution, cmap='jet')
    plt.colorbar()
    plt.title("Aggregated Global SHAP Attribution")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# Inside GlobalShap.py
def measure_avg_time_across_images(data_source, model, generate_attribution, url_data=False):
    """
    Measure average time taken to generate attributions across images for global SHAP.
    
    Parameters:
    - data_source: Data source for images, either a DataLoader or list of images.
    - model: Pretrained model.
    - generate_attribution: Function to generate attributions.
    - url_data: Boolean flag for URL data, which has no labels.

    Returns:
    - Average time taken, list of times per image, representative image, and aggregated SHAP attribution.
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
            representative_image = data_source[2][0].cpu().numpy().squeeze() 

        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)

        # Timing attribution generation
        start_time = timeit.default_timer()
        attribution = generate_attribution(image, predicted_class, model)
        end_time = timeit.default_timer()
        
        # Check and print shapes for debugging
        #print(f"Image {idx + 1} attribution shape before processing: {attribution.shape}")
        
        # Ensure all attributions are consistent in shape
        # Average over the color channels if the shape is (3, 224, 224)
        if attribution.shape == (1, 3, 224, 224):
            attribution = attribution.squeeze()  # Remove batch dimension
            attribution = attribution.mean(axis=0)  # Average across the color channels
        elif attribution.shape != (224, 224):  # Unexpected shape
            raise ValueError(f"Inconsistent shape detected: {attribution.shape} (expected (224, 224))")

        attributions.append(attribution)
        times.append(end_time - start_time)
        torch.cuda.empty_cache()

    avg_time_taken = sum(times) / len(times)
    aggregated_attribution = np.mean(attributions, axis=0)  # Aggregate across all batches
    #print("Shape of aggregated_attribution after averaging:", aggregated_attribution.shape)  # Debugging line
    return avg_time_taken, times, representative_image, aggregated_attribution

import matplotlib.pyplot as plt
import numpy as np

def create_composite_visualization(original_images, aggregated_attribution, visualization_dir, xai_method, model_name, data_type, save_image=True):
    """
    Create a composite visualization with original images placed around and heatmap in the center.

    Parameters:
    - original_images: List of original images to display (in a grid format).
    - aggregated_attribution: The heatmap to overlay.
    - visualization_dir: Directory where the visualization will be saved.
    - xai_method, model_name, data_type: Used for the file name.
    - save_image: Boolean flag to save the generated visualization image.
    """

    # Initialize the figure with a 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    # Add original images around the center (0,0), (0,1), ..., excluding (1,1)
    img_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
    for idx, pos in enumerate(img_positions):
        row, col = pos
        if idx < len(original_images):  # Ensure we don't exceed the number of images
            img = original_images[idx].cpu().numpy().transpose(1, 2, 0)  # Ensure (H, W, C) format
            axs[row, col].imshow(img)
        axs[row, col].axis('off')  # Hide the axis for a cleaner look

    # Place the aggregated attribution heatmap in the center (1,1)
    center_ax = axs[1, 1]
    center_ax.imshow(aggregated_attribution, cmap='viridis', alpha=0.8)
    center_ax.axis('off')  # Hide the axis for the heatmap

    # Add a colorbar for the heatmap
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap='viridis'),
        ax=center_ax, fraction=0.046, pad=0.04
    )
    cbar.set_label("Attribution Intensity", rotation=270, labelpad=15)

    # Save the figure if needed
    if save_image:
        visualization_filename = f"{visualization_dir}/{xai_method}_{model_name}_{data_type}_composite.png"
        plt.savefig(visualization_filename, bbox_inches="tight", pad_inches=0.1)
        print(f"Composite visualization saved as {visualization_filename}")

    plt.show()




# Save a few visualization examples of aggregated attributions
def save_global_attributions_visualizations(aggregated_attribution, xai_method, model_name, data_type):
    visualization_dir = "experiment_results/visualizations"
    os.makedirs(visualization_dir, exist_ok=True)
    visualization_filename = f"{visualization_dir}/{xai_method}_{model_name}_{data_type}_global_shap.png"
    visualize_aggregated_attribution(aggregated_attribution, save_path=visualization_filename)
