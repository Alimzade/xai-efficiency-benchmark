import torch
import time
from PIL import Image
import requests
from io import BytesIO
from captum.attr import Saliency
from models.model_loader import load_model, preprocess_image

def test_run():
    # 1. Determine Device (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"--- Using device: {device} ---")

    # 2. Load Model (ResNet-50)
    print("Loading ResNet-50 model...")
    model = load_model(model_name='resnet50', device=device)
    
    # 3. Load Sample Image from URL
    img_url = "https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg"
    print(f"Downloading sample image from: {img_url}")
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content)).convert('RGB')

    # 4. Preprocess Image
    input_tensor = preprocess_image(img, model_name='resnet50').unsqueeze(0).to(device)

    # 5. Get Prediction
    print("Running inference...")
    with torch.no_grad():
        output = model(input_tensor)
        prediction_score, pred_label_idx = torch.max(output, 1)
        print(f"Predicted class index: {pred_label_idx.item()}")

    # 6. Run XAI Method (Saliency)
    print("Generating XAI explanation (Saliency)...")
    saliency = Saliency(model)
    
    start_time = time.time()
    attribution = saliency.attribute(input_tensor, target=pred_label_idx)
    end_time = time.time()
    
    # 7. Print Results
    elapsed_time = end_time - start_time
    print(f"\n--- SUCCESS! ---")
    print(f"Time taken to generate explanation: {elapsed_time:.4f} seconds")
    print(f"Attribution shape: {attribution.shape}")

if __name__ == "__main__":
    try:
        test_run()
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(e)
        print("\nMake sure you have run: pip install -r requirements.txt")
