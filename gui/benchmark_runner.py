import os
import sys
import time
import json
import torch
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from memory_profiler import memory_usage
import gc
import numpy as np
import matplotlib.pyplot as plt
import platform

# Add the parent directory to sys.path so we can import models and xai_methods
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loader import load_model, preprocess_image
from models.label_utils import get_label_mapping
from captum.attr import Saliency, IntegratedGradients, GuidedBackprop, InputXGradient
from captum.attr import visualization as viz

def run_benchmark_task(config, session_dir):
    """
    Executes a benchmark based on the config and saves results to session_dir.
    Includes memory optimization for high-resolution XAI.
    """
    # 1. Environment Setup for Memory Stability
    # 1. Setup Device & Environment
    force_dev = config.get('force_device')
    device = torch.device(force_dev if force_dev else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Get specific device name for logging
    if device.type == 'cuda':
        device_info = torch.cuda.get_device_name(device)
    else:
        # Use platform info for CPU name
        device_info = platform.processor() or "Generic CPU"

    if torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
        gc.collect()

    # 2. Load Model
    model_name = config.get('model_name', 'resnet50')
    model = load_model(model_name=model_name, device=device)

    # 3. Load Image
    img_src = config.get('image_source')
    target_size = config.get('input_size', 224)
    
    if img_src.startswith('http'):
        response = requests.get(img_src)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(img_src).convert('RGB')
    
    img.save(os.path.join(session_dir, "input_image.jpg"))
    input_tensor = preprocess_image(img, model_name=model_name, target_size=target_size).unsqueeze(0).to(device)
    img_dims = f"{input_tensor.shape[2]} x {input_tensor.shape[3]}"

    # 4. Get Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred_label_idx = torch.max(output, 1)
        predicted_class, _ = get_label_mapping(
            model_name=model_name, predicted_class=pred_label_idx, label=None, label_names=None
        )

    # 5. Benchmarking Loop
    results = []
    methods_to_run = config.get('methods', ['saliency'])
    heatmaps_dir = os.path.join(session_dir, "heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)

    for method_name in methods_to_run:
        try:
            # Clear cache before every method
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            if method_name.lower() == 'saliency': xai_tool = Saliency(model)
            elif method_name.lower() == 'integrated_gradients': xai_tool = IntegratedGradients(model)
            elif method_name.lower() == 'guided_backprop': xai_tool = GuidedBackprop(model)
            elif method_name.lower() == 'input_x_gradient': xai_tool = InputXGradient(model)
            else: continue

            def get_attr():
                if method_name.lower() == 'integrated_gradients':
                    # INTERNAL BATCHING: This is the key to preventing OOM for IG
                    return xai_tool.attribute(input_tensor, target=pred_label_idx, n_steps=50, internal_batch_size=2)
                return xai_tool.attribute(input_tensor, target=pred_label_idx)

            start_time = time.time()
            mem_usage = memory_usage((get_attr, ()), interval=0.1)
            end_time = time.time()
            
            # Generate Overlay
            attribution = get_attr()
            attr_np = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))
            img_resized = np.array(img.resize((target_size, target_size)))
            
            fig, _ = viz.visualize_image_attr(attr_np, img_resized, method="blended_heat_map", sign="all", show_colorbar=True, alpha_overlay=0.6)
            fig.savefig(os.path.join(heatmaps_dir, f"{method_name}.png"), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            results.append({
                "Method": method_name, "Model": model_name, "Resolution": img_dims,
                "Prediction": predicted_class,
                "Device": device_info,
                "Runtime (sec)": round(end_time - start_time, 4),
                "Peak Memory (MB)": round(max(mem_usage) - min(mem_usage), 2)
            })

            # Explicitly delete objects and clear cache after each method
            del attribution, attr_np, xai_tool
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            results.append({
                "Method": method_name, "Model": model_name, "Resolution": img_dims,
                "Prediction": predicted_class if 'predicted_class' in locals() else "N/A",
                "Device": device_info,
                "Runtime (sec)": 0.0, "Peak Memory (MB)": 0.0, "Status": f"Error: {str(e)}"
            })

    # 6. Save Results
    pd.DataFrame(results).to_csv(os.path.join(session_dir, "results.csv"), index=False)
    with open(os.path.join(session_dir, "config.json"), 'w') as f:
        json.dump({**config, "prediction": predicted_class}, f, indent=4)

    # Final cleanup before returning to Streamlit
    del model, input_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    return results
