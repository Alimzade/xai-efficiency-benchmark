import torch
import json
import requests

def get_label_mapping(model_name, predicted_class, label, label_names):
    """
    Get predicted and true labels based on the model name and provided label mappings.

    Parameters:
    - model_name: The name of the model (e.g., 'resnet50', 'convnext-t', 'efficientnet-b0', 'vit-b-16').
    - predicted_class: Predicted class index from the model.
    - label: True label of the image (or None).
    - label_names: List of class names specific to the dataset (or None).

    Returns:
    - predicted_label: Predicted class label (or numerical index if no mapping is found).
    - true_label: True class label (or "Unknown" if no mapping is found).
    """
    # ImageNet label mapping (applicable for most models)
    imagenet_models = [
        'resnet50', 'convnext-t', 'efficientnet-b0',
        'swin-t', 'regnet-y-8gf', 'mobilenet-v3-large',
        'densenet121', 'vit-b-16'  
    ]

    # Fetch ImageNet labels for supported models
    if model_name in imagenet_models:
        try:
            labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
            imagenet_label_names = json.loads(requests.get(labels_url).text)

            # Fetch predicted label
            predicted_label = imagenet_label_names[predicted_class.item()]
        except IndexError:
            predicted_label = f"Class Index {predicted_class.item()}"
        except Exception as e:
            print(f"Error fetching ImageNet labels: {e}")
            predicted_label = f"Class Index {predicted_class.item()}"
    else:
        # Handle unsupported models or custom datasets
        print(
            f"Warning: Label mapping for model '{model_name}' is missing. "
            f"Please add the mapping for this model in 'label_utils.py'."
        )
        try:
            predicted_label = label_names[predicted_class.item()]
        except (IndexError, TypeError):
            predicted_label = f"Class Index {predicted_class.item()}"

    # Fetch true label
    if label_names and label is not None:
        try:
            true_label = label_names[label.item()] if isinstance(label, torch.Tensor) else label_names[label]
        except IndexError:
            true_label = f"Class Index {label.item()}"
    else:
        true_label = "Unknown"

    return predicted_label, true_label
