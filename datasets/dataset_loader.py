from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from PIL import Image
import requests
import os
from io import BytesIO

# Transformation for model input (resize for models like ResNet and ViT)
transform_model_input = transforms.Compose([
    transforms.Resize((224, 224)),  # Force resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Transformation for LIME (also force resize to 224x224)
transform_no_resize = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_data(data_type, num_images, transform=None, image_urls=None, selected_class=None):
    """
    Load a specific data source based on `data_type`, using a passed `transform`.

    Parameters:
    - data_type: Type of dataset to load ('CIFAR10', 'STL10', or 'URL').
    - num_images: Number of images to load for benchmarking.
    - transform: The transformation function to apply.
    - image_urls: List of image URLs if `data_type` is 'URL'.
    - selected_class: Optional class index to filter by if using a global method.

    Returns:
    - Data source and class labels (or placeholder labels for URLs).
    """
    if data_type == 'CIFAR10':
        dataset = datasets.CIFAR10(root='./datasets/data', train=False, transform=transform, download=True)
        if selected_class is not None:
            indices = [i for i, (_, label) in enumerate(dataset) if label == selected_class]
            subset = Subset(dataset, indices[:num_images])
        else:
            subset = Subset(dataset, list(range(num_images)))
        return DataLoader(subset, batch_size=1, shuffle=False), dataset.classes

    elif data_type == 'STL10':
        dataset = datasets.STL10(root='./datasets/data', split='test', transform=transform, download=True)
        if selected_class is not None:
            indices = [i for i, (_, label) in enumerate(dataset) if label == selected_class]
            subset = Subset(dataset, indices[:num_images])
        else:
            subset = Subset(dataset, list(range(num_images)))
        return DataLoader(subset, batch_size=1, shuffle=False), dataset.classes

    elif data_type == 'ImageNet':
        imagenet_dir = './datasets/data/imagenet_val/ILSVRC2012'
        if not os.path.exists(imagenet_dir):
            raise ValueError(f"ImageNet validation directory '{imagenet_dir}' does not exist. Please extract the tarball.")
        dataset = ImageFolder(
            root=imagenet_dir,
            transform=transform,
            is_valid_file=lambda path: path.lower().endswith('.jpeg')
        )
        if len(dataset) == 0:
            raise ValueError("No valid images found in the dataset. Please check the directory structure.")
        if num_images is None:
            subset = dataset
        else:
            subset = Subset(dataset, list(range(min(len(dataset), num_images))))
        return DataLoader(subset, batch_size=1, shuffle=False), dataset.classes

    elif data_type == 'URL':
        images = []
        for url in image_urls[:num_images]:
            try:
                print(f"Attempting to load image from: {url}")
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content)).convert('RGB')
                # Apply the provided transform (which must force resize to 224x224)
                images.append(transform(img).unsqueeze(0))
                print(f"Successfully loaded image from: {url}")
            except requests.RequestException as e:
                print(f"Request failed for {url}: {e}")
            except IOError as e:
                print(f"Image processing failed for {url}: {e}")
    
        if len(images) == 0:
            raise ValueError("No images were loaded from the provided URLs.")
    
        labels = [None] * len(images)
    
        return images, labels
