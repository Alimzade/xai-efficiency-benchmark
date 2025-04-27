import requests
import tarfile
import os

def download_and_extract_imagenet_val():
    """
    Download and extract only the ImageNet-1k validation split from Hugging Face using authentication.
    """
    # URL for validation images tarball
    val_images_url = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data/val_images.tar.gz?download=true"
    
    # Directory to save the downloaded and extracted files
    output_dir = "./data/imagenet_val"
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to save the downloaded tarball
    tarball_path = os.path.join(output_dir, "val_images.tar.gz")
    
    # Hugging Face API token (Get it from https://huggingface.co/settings/tokens)
    hf_token = "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"  # Replace with your token
    
    # Headers with authorization
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    # Download the tarball
    print(f"Downloading ImageNet validation tarball from {val_images_url}...")
    response = requests.get(val_images_url, headers=headers, stream=True)
    
    if response.status_code == 200:
        with open(tarball_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        
        # Extract the tarball
        print("Extracting the tarball...")
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
        print(f"Extraction complete. Validation images saved to: {output_dir}")
        
        # Optional: Remove the tarball to save space
        os.remove(tarball_path)
    else:
        print(f"Failed to download the file. HTTP Status Code: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    download_and_extract_imagenet_val()
