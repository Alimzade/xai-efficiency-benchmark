import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from captum.attr import Lime
from utils.data_utils import process_data_tuple
from models.label_utils import get_label_mapping
import timeit


def generate_attribution(image, predicted_class, model):
    """
    Generate LIME attribution for the given image and class.
    
    Parameters:
    - image: Tensor representing the input image.
    - predicted_class: Predicted class label for which we generate the attribution.
    - model: Pretrained model to be used with LIME.
    
    Returns:
    - attribution: LIME attribution for the input image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert the tensor image back to numpy for segmentation
    img_np = image.detach().squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
    
    # Perform superpixel segmentation using SLIC
    superpixels = slic(img_np)
    superpixels = superpixels - superpixels.min()  # Ensure the minimum value in the feature mask is 0
    superpixels = torch.tensor(superpixels, dtype=torch.long).unsqueeze(0).to(device)

    # Initialize LIME
    lime = Lime(model)

    # Generate attributions
    attribution = lime.attribute(image, target=predicted_class.item(), feature_mask=superpixels, n_samples=100)
    attribution = attribution.mean(dim=1, keepdim=False)  # Aggregate attributions across channels
    return attribution


def warm_up(model):
    """
    Run a warm-up pass to ensure memory and computation stability.
    """
    dummy_image = torch.randn(1, 3, 224, 224, device=next(model.parameters()).device, requires_grad=True)
    with torch.no_grad():
        output = model(dummy_image)
        _, predicted_class = torch.max(output, 1)
    _ = generate_attribution(dummy_image, predicted_class, model)


def visualize_attribution(image, attribution, label, label_names, model, model_name, save_path=None):
    """
    Visualize the LIME attribution overlayed on the original image, including predicted class.

    Parameters:
    - image: Original image tensor.
    - attribution: Attribution data to visualize.
    - label: True label of the image (or None for URL data).
    - label_names: List of class names specific to the dataset (or None for URL data).
    - model: Pretrained model used for predictions.
    - model_name: The name of the model (e.g., 'resnet50', 'convnext-t', 'efficientnet-b0').
    - save_path: Path to save the visualization.
    """
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

    # Normalize the attribution
    attribution = attribution.squeeze().cpu().detach().numpy()
    attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)

    # Apply colormap for visualization
    attribution_colored = cv2.applyColorMap((attribution * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

    # Convert the original image tensor to numpy
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)

    # Overlay heatmap on the original image
    overlayed_img = cv2.addWeighted(img_np, 0.6, attribution_colored, 0.4, 0)

    # Plot images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.5)

    axes[0].imshow(img_np)
    axes[0].set_title(f'True: {true_label}' if true_label != "Unknown" else 'Input')
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


# Initialize save counter
visualize_attribution.save_count = 0


def measure_avg_time_across_images(data_source, model, generate_attribution):
    """
    Measure average time taken to generate attributions across images.
    """
    times = []

    for data_tuple in data_source:
        image, label = process_data_tuple(data_tuple, model)
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
