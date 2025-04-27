from datasets.dataset_loader import load_data

def download_datasets():
    """
    Download CIFAR-10 and STL-10 datasets to ensure they're available locally.
    """
    print("Downloading CIFAR-10 dataset...")
    load_data('CIFAR10', num_images=1)  # Only load 1 image to trigger the download if necessary
    
    print("Downloading STL-10 dataset...")
    load_data('STL10', num_images=1)  # Only load 1 image to trigger the download if necessary
    
    print("Datasets are downloaded and ready in './datasets/data'.")

if __name__ == "__main__":
    download_datasets()