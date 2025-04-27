import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import IntegratedGradients
from utils.data_utils import process_data_tuple
from models.label_utils import get_label_mapping
import timeit


def generate_attribution(image, predicted_class, model, baseline=None):
    """
    Generate Integrated Gradients attribution for the given image and class.
    
    Parameters:
    - image: Tensor representing the input image.
    - predicted_class: Predicted class label for which we generate the attribution.
    - model: Pretrained model to be used with Integrated Gradients.
    - baseline: Baseline tensor for Integrated Gradients (default is black image).
    
    Returns:
    - attribution: Aggregated attribution (2D array) across channels.
    """
    # Initialize Integrated Gradients method
    integrated_gradients = IntegratedGradients(model)

    # If no baseline is provided, use a black image baseline (zeros)
    if baseline is None:
        baseline = torch.zeros_like(image).to(image.device)

    # Generate attributions for the target class using Integrated Gradients
    attribution = integrated_gradients.attribute(image, baselines=baseline, target=predicted_class.item())
    
    # Aggregate the attributions across color channels to enhance interpretability
    attribution = attribution.squeeze().cpu().detach().numpy()
    attribution = np.mean(np.abs(attribution), axis=0)  # Aggregating across color channels
    return attribution


def warm_up(model):
    """
    Run a warm-up pass to ensure memory and computation stability.
    
    Parameters:
    - model: Pretrained model to be used with Integrated Gradients.
    """
    # Creating a dummy image on the correct device for the model and requiring gradients
    dummy_image = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device, requires_grad=True)
    
    # Perform a single forward and attribution pass to stabilize
    with torch.no_grad():
        output = model(dummy_image)
        _, predicted_class = torch.max(output, 1)
    _ = generate_attribution(dummy_image, predicted_class, model)


def visualize_attribution(image, attribution, label, label_names, model, model_name, save_path=None):
    """
    Visualize the Integrated Gradients heatmap overlayed on the original image, including predicted class.

    Parameters:
    - image: Original input image tensor.
    - attribution: Integrated Gradients attribution.
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

    # Normalize the attribution data for visualization
    attribution = attribution - attribution.min()
    epsilon = 1e-8  # Small constant to avoid division by zero
    attribution = attribution / (attribution.max() + epsilon)  # Scale to [0, 1]

    # Apply gamma correction to enhance visibility
    gamma = 0.5  # Adjust gamma for contrast; higher values increase contrast
    attribution = np.power(attribution, gamma)
    attribution = (attribution * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    
    # Apply colormap for better visualization
    attribution_colored = cv2.applyColorMap(attribution, cv2.COLORMAP_JET)

    # Convert the original image tensor to a numpy array for visualization
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)
    
    # Resize the heatmap to match the original image size
    heatmap_resized = cv2.resize(attribution_colored, (img_np.shape[1], img_np.shape[0]))

    # Overlay heatmap on the original image for combined visualization
    overlayed_img = cv2.addWeighted(img_np, 0.6, heatmap_resized, 0.4, 0)

    # Plot and optionally save the visualizations
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Increased size for spacing
    fig.subplots_adjust(wspace=0.5)  # Add spacing between subplots

    # Display the original image
    axes[0].imshow(img_np)
    if true_label == "Unknown":
        axes[0].set_title(f'Input')
    else:
        axes[0].set_title(f'True: {true_label}')
    axes[0].axis('off')

    # Display the overlayed heatmap
    axes[1].imshow(overlayed_img)
    axes[1].set_title(f'Predicted: {predicted_label}')
    axes[1].axis('off')

    fig.tight_layout()

    # Save the figure if `save_path` is provided and save count is below the limit
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        visualize_attribution.save_count += 1  # Increment the save counter only if saved
        print(f"Saved visualization at {save_path}")

    plt.show()  # Always display the figure in the notebook
    plt.close(fig)  # Close the figure after showing to release memory


# Initialize the save counter as an attribute of the function
visualize_attribution.save_count = 0


def measure_avg_time_across_images(data_source, model, generate_attribution):
    """
    Measure average time taken to generate attributions across images.
    
    Parameters:
    - data_source: Data source for images, either a DataLoader or list of images.
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
