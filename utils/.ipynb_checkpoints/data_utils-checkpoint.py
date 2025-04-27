import torch

def process_data_tuple(data_tuple, model):
    """
    Process a data tuple (image, label) by moving the image to the model's device.

    Parameters:
    - data_tuple: A tuple containing (image, label).
    - model: The model to determine the device.

    Returns:
    - image: The image tensor moved to the correct device.
    - label: The label (unchanged).
    """
    image, label = data_tuple
    image = image.to(next(model.parameters()).device)
    return image, label