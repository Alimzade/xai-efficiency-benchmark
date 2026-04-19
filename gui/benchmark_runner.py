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

# Add the parent directory to sys.path so we can import models and xai_methods
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loader import load_model, preprocess_image
from models.label_utils import get_label_mapping # USE PROJECT NATIVE LOGIC
from captum.attr import Saliency, IntegratedGradients, GuidedBackprop, InputXGradient

import numpy as np
from captum.attr import visualization as viz

def run_benchmark_task(config, session_dir):
    """
    Executes a benchmark based on the config and saves results to session_dir.
    """
    # 1. Setup Device
    force_dev = config.get('force_device')
    device = torch.device(force_dev if force_dev else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # 2. Load Model
    model_name = config.get('model_name', 'resnet50')
    model = load_model(model_name=model_name, device=device)

    # 3. Load Image
    img_src = config.get('image_source')
    if img_src.startswith('http'):
        response = requests.get(img_src)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(img_src).convert('RGB')
    
    img_dims = f"{img.size[0]} x {img.size[1]}"
    img.save(os.path.join(session_dir, "input_image.jpg"))
    input_tensor = preprocess_image(img, model_name=model_name).unsqueeze(0).to(device)

    # 4. Get Prediction using PROJECT NATIVE label_utils
    with torch.no_grad():
        output = model(input_tensor)
        _, pred_label_idx = torch.max(output, 1)
        
        # Call your original mapping function
        predicted_class, _ = get_label_mapping(
            model_name=model_name,
            predicted_class=pred_label_idx,
            label=None, # True label unknown for batch uploads/URLs
            label_names=None
        )

    # 5. Benchmarking Loop
    results = []
    methods_to_run = config.get('methods', ['saliency'])
    heatmaps_dir = os.path.join(session_dir, "heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)

    for method_name in methods_to_run:
        try:
            # Map method name
            if method_name.lower() == 'saliency': xai_tool = Saliency(model)
            elif method_name.lower() == 'integrated_gradients': xai_tool = IntegratedGradients(model)
            elif method_name.lower() == 'guided_backprop': xai_tool = GuidedBackprop(model)
            elif method_name.lower() == 'input_x_gradient': xai_tool = InputXGradient(model)
            else: continue

            def get_attr():
                if method_name.lower() == 'integrated_gradients':
                    return xai_tool.attribute(input_tensor, target=pred_label_idx, n_steps=50)
                return xai_tool.attribute(input_tensor, target=pred_label_idx)

            start_time = time.time()
            mem_usage = memory_usage((get_attr, ()), interval=0.1)
            end_time = time.time()
            
            # Generate Overlay
            attribution = get_attr()
            attr_np = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))
            img_resized = np.array(img.resize((224, 224)))
            
            fig, _ = viz.visualize_image_attr(attr_np, img_resized, method="blended_heat_map", sign="all", show_colorbar=True, alpha_overlay=0.6)
            fig.savefig(os.path.join(heatmaps_dir, f"{method_name}.png"), bbox_inches='tight', pad_inches=0)
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            results.append({
                "Method": method_name, "Model": model_name, "Resolution": img_dims,
                "Prediction": predicted_class,
                "Runtime (sec)": round(end_time - start_time, 4),
                "Peak Memory (MB)": round(max(mem_usage) - min(mem_usage), 2)
            })
        except Exception as e:
            results.append({"Method": method_name, "Model": model_name, "Prediction": predicted_class, "Runtime (sec)": 0.0, "Peak Memory (MB)": 0.0, "Status": f"Error"})

    # 6. Save Results
    pd.DataFrame(results).to_csv(os.path.join(session_dir, "results.csv"), index=False)
    with open(os.path.join(session_dir, "config.json"), 'w') as f:
        json.dump({**config, "prediction": predicted_class}, f, indent=4)

    return results
