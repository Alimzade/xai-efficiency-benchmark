import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.data_utils import process_data_tuple
from models.label_utils import get_label_mapping
import timeit

def generate_rise_masks(image_shape, num_masks=4000, mask_size=14, device=None):
    """
    Generate a batch of random binary masks for RISE.
    """
    h, w = image_shape[-2], image_shape[-1]
    masks = np.random.randint(0, 2, size=(num_masks, mask_size, mask_size)).astype(np.float32)
    resized_masks = [cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) for mask in masks]
    stacked_masks = np.stack(resized_masks)
    return torch.tensor(stacked_masks, device=device)


def generate_attribution(image, predicted_class, model, num_masks=4000):
    """
    Generate RISE attributions by applying random masks.
    """
    device = image.device  # Ensure the masks are on the same device as the image
    masks = generate_rise_masks(image.shape, num_masks=num_masks, device=device)
    scores = torch.zeros(masks.shape[0], device=device)  # Initialize scores on the same device

    for i, mask in enumerate(masks):
        masked_image = image * mask.unsqueeze(0)  # Apply mask to the image
        with torch.no_grad():
            output = model(masked_image)
            prob = torch.nn.functional.softmax(output, dim=1)[0, predicted_class]
            scores[i] = prob

    attribution = (masks * scores.view(-1, 1, 1)).mean(0)  # Average weighted masks
    return attribution


def warm_up(model):
    """
    Run a warm-up pass to ensure memory and computation stability.
    """
    device = next(model.parameters()).device
    dummy_image = torch.zeros(1, 3, 224, 224, device=device)
    
    with torch.no_grad():
        output = model(dummy_image)
        _, predicted_class = torch.max(output, 1)

    dummy_masks = generate_rise_masks(dummy_image.shape, num_masks=10, device=device)
    for mask in dummy_masks:
        masked_image = dummy_image * mask.unsqueeze(0)
        _ = model(masked_image)


def visualize_attribution(image, attribution, label, label_names, model, model_name, save_path=None):
    """
    Visualize the RISE attribution overlayed on the original image.
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

    # Process attribution for visualization
    attribution = attribution.cpu().numpy()
    attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())
    threshold = np.percentile(attribution, 85)
    attribution[attribution < threshold] = 0
    attribution = (attribution * 255).astype(np.uint8)
    attribution_colored = cv2.applyColorMap(attribution, cv2.COLORMAP_JET)

    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip((img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)
    heatmap_resized = cv2.resize(attribution_colored, (img_np.shape[1], img_np.shape[0]))
    overlayed_img = cv2.addWeighted(img_np, 0.6, heatmap_resized, 0.4, 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
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
        _ = generate_attribution(image, predicted_class.item(), model)
        end_time = timeit.default_timer()

        times.append(end_time - start_time)
        torch.cuda.empty_cache()

    avg_time_taken = sum(times) / len(times)
    return avg_time_taken, times
