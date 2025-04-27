import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import slic
from captum.attr import Lime
import timeit

def generate_attribution(image, predicted_class, model, n_segments, n_samples=100, compactness=10.0, sigma=1.0):
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
    
    # Perform superpixel segmentation using SLIC (default parameters)
    superpixels = slic(img_np, n_segments=n_segments, compactness=compactness, sigma=sigma)
    
    # Ensure the minimum value in the feature mask is 0
    superpixels = superpixels - superpixels.min()
    
    # Convert superpixels to tensor and reshape for Lime
    superpixels = torch.tensor(superpixels, dtype=torch.long).unsqueeze(0).to(device)

    # Initialize LIME for this specific model instance
    lime = Lime(model)

    # Generate attributions using LIME with the superpixel mask
    attribution = lime.attribute(image, target=predicted_class.item(), feature_mask=superpixels, n_samples=n_samples)
    
    # Aggregate the attribution across channels to get a 2D array
    attribution = attribution.mean(dim=1, keepdim=False)  # Take the mean across the channel dimension

    # Normalize the attribution map (following the same approach as visualization)
    attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
    return attribution



def warm_up(model):
    """
    Run a warm-up pass to ensure memory and computation stability.
    
    Parameters:
    - model: Pretrained model to be used with LIME.
    """
    dummy_image = torch.randn(1, 3, 32, 32, device=next(model.parameters()).device, requires_grad=True)
    with torch.no_grad():
        output = model(dummy_image)
        _, predicted_class = torch.max(output, 1)
    _ = generate_attribution(dummy_image, predicted_class, model, n_segments=50)


# Function to denormalize images
def denormalize(image_tensor, mean, std):
    image_tensor = image_tensor.clone()
    for c in range(3):  # Assuming RGB
        image_tensor[c] = image_tensor[c] * std[c] + mean[c]
    return image_tensor

def visualize_attribution(image, attribution, label, label_names, save_path=None):
    """
    Visualize the original image alongside the LIME attribution heatmap.

    Parameters:
    - image: Original image tensor.
    - attribution: Attribution data to visualize.
    - label: Predicted label or actual label for the image.
    - label_names: List of label names (class names).
    - save_path: Path to save the visualization.
    - save_limit: Maximum number of images to save.
    """
    # Track the number of saved images
    if visualize_attribution.save_count >= 3:
        save_path = None  # Disable saving if the limit is reached

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Denormalize the original image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image_denorm = denormalize(image.squeeze().cpu(), mean, std)
    img_np = image_denorm.numpy().transpose(1, 2, 0).clip(0, 1)  # Ensure values are in [0, 1] for display

    # Plot original image on the left
    axes[0].imshow(img_np)
    if label_names and label is not None:
        title_label = label_names[label.item()] if isinstance(label, torch.Tensor) else label_names[label]
        axes[0].set_title(f'Original Image, Label: {title_label}')
    else:
        axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Move attribution tensor to CPU and convert to NumPy
    attribution = attribution.cpu().detach().numpy()

    # Apply thresholding to only keep the top 10% most important superpixels in the attribution
    top_superpixels = np.percentile(attribution, 90)  # Threshold to show only top 10%
    attribution[attribution < top_superpixels] = 0  # Zero out less important superpixels

    # Ensure attribution is single-channel and of type uint8
    attribution = (attribution * 255).astype(np.uint8)
    if attribution.ndim > 2:  # Reduce to single channel if multi-channel
        attribution = np.mean(attribution, axis=0).astype(np.uint8)

    # Apply colormap to attribution
    attribution_colored = cv2.applyColorMap(attribution, cv2.COLORMAP_VIRIDIS)

    # Resize heatmap and overlay on the original image size
    heatmap_resized = cv2.resize(attribution_colored, (img_np.shape[1], img_np.shape[0]))
    overlayed_img = cv2.addWeighted((img_np * 255).astype(np.uint8), 0.6, heatmap_resized, 0.4, 0)

    # Plot the overlay image on the right
    axes[1].imshow(overlayed_img)
    axes[1].set_title('LIME Attribution (Top Superpixels)')
    axes[1].axis('off')

    # Save the image if save_path is provided and within the save limit
    if save_path:
        plt.savefig(save_path)
        visualize_attribution.save_count += 1  # Increment the counter only if saved
        print(f"Saved visualization at {save_path}")

    plt.show()
    plt.close(fig)  # Close the figure after showing to release memory


# Initialize the save counter as an attribute of the function
visualize_attribution.save_count = 0





def measure_avg_time_across_images(data_source, model, generate_attribution, n_segments, n_samples, compactness, sigma):
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

    # Check if data_source is a DataLoader (has an iterator)
    if isinstance(data_source, torch.utils.data.DataLoader):
        for images, labels in data_source:
            image = images.to(next(model.parameters()).device)
            with torch.no_grad():
                output = model(image)
                _, predicted_class = torch.max(output, 1)

            # Measure time for attribution
            start_time = timeit.default_timer()
            _ = generate_attribution(image, predicted_class, model,
                             n_segments=n_segments, n_samples=n_samples,
                             compactness=compactness, sigma=sigma)
            end_time = timeit.default_timer()

            times.append(end_time - start_time)
            torch.cuda.empty_cache()

    else:  # Assuming data_source is a list of preprocessed images (e.g., CIFAR-10)
        for image in data_source:
            image = image.to(next(model.parameters()).device)
            with torch.no_grad():
                output = model(image)
                _, predicted_class = torch.max(output, 1)

            # Measure time for attribution
            start_time = timeit.default_timer()
            _ = generate_attribution(image, predicted_class, model,
                             n_segments=n_segments, n_samples=n_samples,
                             compactness=compactness, sigma=sigma)
            end_time = timeit.default_timer()

            times.append(end_time - start_time)
            torch.cuda.empty_cache()

    avg_time_taken = sum(times) / len(times)
    return avg_time_taken, times