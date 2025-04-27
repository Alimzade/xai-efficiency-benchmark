from datasets.dataset_loader import load_data, transform_no_resize, transform_model_input
from torch.utils.data import DataLoader, Subset
import random
from collections import Counter

def load_dataset(data_type, num_images, image_urls=None, selected_class=None, transform=None, is_global_method=False):
    """
    Optimized function to handle dataset loading for various data types with random sampling and class counts.
    """
    if data_type == 'URL':
        print("Loading URL data...")
        num_images = len(image_urls)
        if is_global_method:
            data_source, labels = load_data(
                data_type=data_type, num_images=num_images, image_urls=image_urls, 
                selected_class=selected_class, transform=transform
            )
        else:
            data_source, labels = load_data(
                data_type=data_type, num_images=num_images, image_urls=image_urls, transform=transform
            )
        print(f"Loaded {len(data_source)} images from URLs.")

        # Ensure every image has a tuple with None label if labels are missing
        data_source = [(image, None) for image in data_source]
        labels = None
        
    elif data_type == 'ImageNet':
        print("Loading ImageNet validation data...")
        data_loader, classes = load_data(
            data_type='ImageNet', num_images=None, transform=transform
        )
        
        dataset_size = len(data_loader.dataset)
        indices = random.sample(range(dataset_size), min(num_images, dataset_size))
        subset = Subset(data_loader.dataset, indices)
        subset_loader = DataLoader(subset, batch_size=1)

        final_images, final_labels = [], []
        for image, label in subset_loader:
            final_images.append(image)
            final_labels.append(label.item())  # Store label indices

        # Combine images and labels into a list of tuples
        data_source = list(zip(final_images[:num_images], final_labels[:num_images]))

        # Print class counts
        class_counts = Counter(final_labels)
        #print("Loaded ImageNet images per class:")
        #for class_name, count in class_counts.items():
            #print(f"  {class_name}: {count} images")

        # Ensure labels is defined
        labels = classes
        
    elif data_type in ['CIFAR10', 'STL10']:
        print(f"Loading {data_type} data with random sampling...")

        # Load dataset
        data_loader, _ = load_data(
            data_type=data_type, num_images=num_images, transform=transform
        )

        # Randomly sample images
        dataset_size = len(data_loader.dataset)
        indices = random.sample(range(dataset_size), min(num_images, dataset_size))
        subset = Subset(data_loader.dataset, indices)
        subset_loader = DataLoader(subset, batch_size=1)

        data_source = [(image, label.item()) for image, label in subset_loader]

        # No need to map labels for CIFAR10 and STL10
        labels = None

    else:
        raise ValueError(f"Unsupported dataset type: {data_type}")

    return data_source, labels