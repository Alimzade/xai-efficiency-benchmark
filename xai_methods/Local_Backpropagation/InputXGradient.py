import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import InputXGradient
from utils.data_utils import process_data_tuple
from models.label_utils import get_label_mapping
import timeit


def generate_attribution(image, predicted_class, model):
    """
    Generate Input X Gradient attribution for the given image and class.
    
    Parameters:
    - image: Tensor representing the input image.
    - predicted_class: Predicted class label for which we generate the attribution.
    - model: Pretrained model to be used with Input X Gradient.
    """
    # Initialize Input X Gradient
    input_x_gradient = InputXGradient(model)

    # Generate attributions
    attribution = input_x_gradient.attribute(image, target=predicted_class.item())
    return attribution


def warm_up(model):
    """
    Run a warm-up pass to ensure memory and computation stability.
    
    Parameters:
    - model: Pretrained model to be used with Input X Gradient.
    """
    dummy_image = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device, requires_grad=True)
    with torch.no_grad():
        output = model(dummy_image)
        _, predicted_class = torch.max(output, 1)
    _ = generate_attribution(dummy_image, predicted_class, model)


def visualize_attribution(image, attribution, label, label_names, model, model_name, save_path=None):
    """
    Visualize the Input X Gradient heatmap overlayed on the original image, including predicted class.

    Parameters:
    - image: Original input image tensor.
    - attribution: Input X Gradient attribution.
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

    # Convert Input X Gradient attribution to numpy array and normalize
    attribution = attribution.squeeze().cpu().detach().numpy()
    attribution = np.abs(attribution)
    if attribution.ndim > 2:
        attribution = np.mean(attribution, axis=0)  # Reduce to single channel
    attribution = np.maximum(attribution, 0)
    if attribution.max() > 0:
        attribution /= attribution.max()

    # Apply colormap
    attribution_colored = cv2.applyColorMap((attribution * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Convert the image tensor to a numpy array for visualization
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)
    
    # Resize the heatmap to match the original image size
    heatmap_resized = cv2.resize(attribution_colored, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)
    overlayed_img = cv2.addWeighted(img_np, 0.6, heatmap_resized, 0.4, 0)

    # Create the figure and axes for plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Increased size for spacing
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

    # Save the figure if save_path is provided
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        visualize_attribution.save_count += 1
        print(f"Saved visualization at {save_path}")

    plt.show()
    plt.close(fig)


# Initialize the save counter as an attribute of the function
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
