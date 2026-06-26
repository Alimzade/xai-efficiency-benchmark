import os
import sys
import time
import json
import subprocess
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
from datetime import datetime

# Add the parent directory to sys.path so we can import models and xai_methods
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_loader import load_model, preprocess_image
from models.label_utils import get_label_mapping
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    GradientShap,
    GuidedBackprop,
    InputXGradient,
    IntegratedGradients,
    LayerAttribution,
    LayerGradCam,
    Saliency,
)
from captum.attr import visualization as viz

MODEL_CACHE = {}
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRAD_CAM_TARGET_LAYERS = {
    "resnet50": lambda model: model.layer4[-1],
    "convnext-t": lambda model: model.features[-1],
    "efficientnet-b0": lambda model: model.features[-1],
    "regnet-y-8gf": lambda model: model.trunk_output.block4,
    "mobilenet-v3-large": lambda model: model.features[-1],
    "densenet121": lambda model: model.features.denseblock4,
}

def sync_device(device):
    """Wait for queued CUDA/MPS work so wall-clock timing reflects actual GPU work."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

def get_cached_model(model_name, device):
    cache_key = (model_name, str(device))
    
    # Keep at most 2 different models in cache to prevent VRAM accumulation OOM
    if cache_key not in MODEL_CACHE and len(MODEL_CACHE) >= 2:
        oldest_key = list(MODEL_CACHE.keys())[0]
        del MODEL_CACHE[oldest_key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        gc.collect()

    was_cached = cache_key in MODEL_CACHE
    if not was_cached:
        MODEL_CACHE[cache_key] = load_model(model_name=model_name, device=device)
    return MODEL_CACHE[cache_key], was_cached

def get_grad_cam_target_layer(model_name, model):
    if model_name not in GRAD_CAM_TARGET_LAYERS:
        raise ValueError(f"Grad_CAM is not configured for model '{model_name}'.")
    return GRAD_CAM_TARGET_LAYERS[model_name](model)

def normalize_method_name(method_name):
    return method_name.lower().replace("-", "_")

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        return "unknown"

def collect_environment_metadata(device=None):
    cuda_devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            cuda_devices.append({
                "index": idx,
                "name": torch.cuda.get_device_name(idx),
                "total_memory_mb": round(props.total_memory / (1024 * 1024), 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })

    return {
        "app_version": "1.0.0",
        "git_commit": get_git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "Generic CPU",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_devices": cuda_devices,
        "selected_device": str(device) if device is not None else None,
    }

def run_benchmark_task(config, session_dir):
    """
    Executes a benchmark based on the config and saves results to session_dir.
    Includes memory optimization for high-resolution XAI.
    """
    task_started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    # 1. Environment Setup for Memory Stability
    # 1. Setup Device & Environment
    force_dev = config.get('force_device')
    device = torch.device(force_dev if force_dev else ("cuda" if torch.cuda.is_available() else "cpu"))
    environment_metadata = collect_environment_metadata(device)

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
    model, was_model_cached = get_cached_model(model_name=model_name, device=device)

    # 3. Load Image
    img_src = config.get('image_source')
    target_size = config.get('input_size', 224)
    
    if img_src.startswith('http'):
        response = requests.get(img_src)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(img_src).convert('RGB')
    
    original_dims = f"{img.size[0]} x {img.size[1]}"
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
    warmup_runs = max(0, int(config.get('warmup_runs', 1)))
    repeat_count = max(1, int(config.get('repeat_count', 1)))
    heatmaps_dir = os.path.join(session_dir, "heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)

    for method_name in methods_to_run:
        try:
            method_key = normalize_method_name(method_name)
            # Clear cache before every method
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            if method_key == 'saliency': xai_tool = Saliency(model)
            elif method_key == 'integrated_gradients': xai_tool = IntegratedGradients(model)
            elif method_key == 'guided_backprop': xai_tool = GuidedBackprop(model)
            elif method_key == 'input_x_gradient': xai_tool = InputXGradient(model)
            elif method_key == 'gradient_shap': xai_tool = GradientShap(model)
            elif method_key == 'deeplift': xai_tool = DeepLift(model)
            elif method_key == 'deeplift_shap': xai_tool = DeepLiftShap(model)
            elif method_key == 'grad_cam': xai_tool = LayerGradCam(model, get_grad_cam_target_layer(model_name, model))
            else: continue

            def get_attr():
                if method_key == 'integrated_gradients':
                    # INTERNAL BATCHING: This is the key to preventing OOM for IG
                    return xai_tool.attribute(input_tensor, target=pred_label_idx, n_steps=50, internal_batch_size=2)
                if method_key == 'gradient_shap':
                    baseline_dist = torch.cat([torch.zeros_like(input_tensor), torch.ones_like(input_tensor) * input_tensor.mean()], dim=0)
                    return xai_tool.attribute(input_tensor, baselines=baseline_dist, target=pred_label_idx, n_samples=10, stdevs=0.0001)
                if method_key == 'deeplift':
                    return xai_tool.attribute(input_tensor, baselines=torch.zeros_like(input_tensor), target=pred_label_idx)
                if method_key == 'deeplift_shap':
                    baseline_dist = torch.cat([torch.zeros_like(input_tensor), torch.ones_like(input_tensor) * input_tensor.mean()], dim=0)
                    return xai_tool.attribute(input_tensor, baselines=baseline_dist, target=pred_label_idx)
                if method_key == 'grad_cam':
                    attribution = xai_tool.attribute(input_tensor, target=pred_label_idx)
                    attribution = LayerAttribution.interpolate(attribution, input_tensor.shape[2:])
                    return attribution.repeat(1, 3, 1, 1)
                return xai_tool.attribute(input_tensor, target=pred_label_idx)

            def timed_get_attr(measure_memory=True):
                peak_memory_mb = None
                if measure_memory:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
                        torch.cuda.reset_peak_memory_stats(device)
                        memory_before = torch.cuda.memory_allocated(device)
                    elif device.type == 'mps':
                        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        gc.collect()
                        memory_before = torch.mps.current_allocated_memory() if (hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory')) else 0

                sync_device(device)
                start_time = time.perf_counter()
                attribution_result = get_attr()
                sync_device(device)
                runtime_sec = time.perf_counter() - start_time

                if measure_memory:
                    if device.type == 'cuda':
                        peak_memory = torch.cuda.max_memory_allocated(device)
                        peak_memory_mb = max(peak_memory - memory_before, 0) / (1024 * 1024)
                    elif device.type == 'mps':
                        memory_after = torch.mps.current_allocated_memory() if (hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory')) else 0
                        peak_memory_mb = max(memory_after - memory_before, 0) / (1024 * 1024)

                return attribution_result, runtime_sec, peak_memory_mb

            for _ in range(warmup_runs):
                warmup_attribution, _, _ = timed_get_attr(measure_memory=False)
                del warmup_attribution
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                gc.collect()

            attribution = None
            runtime_values = []
            memory_values = []

            for _ in range(repeat_count):
                if device.type in ['cuda', 'mps']:
                    current_attribution, current_runtime, current_memory = timed_get_attr()
                else:
                    mem_usage, timed_result = memory_usage((timed_get_attr, ()), interval=0.1, retval=True)
                    current_attribution, current_runtime, _ = timed_result
                    current_memory = max(mem_usage) - min(mem_usage) if mem_usage else 0.0

                if attribution is not None:
                    del attribution
                attribution = current_attribution
                runtime_values.append(current_runtime)
                memory_values.append(current_memory if current_memory is not None else 0.0)

            runtime_median = float(np.median(runtime_values))
            runtime_mean = float(np.mean(runtime_values))
            runtime_std = float(np.std(runtime_values))
            runtime_min = float(np.min(runtime_values))
            runtime_max = float(np.max(runtime_values))
            peak_memory_mb = float(max(memory_values)) if memory_values else 0.0
            
            # Generate Overlay
            attr_np = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))
            img_resized = np.array(img.resize((target_size, target_size)))
            
            fig, _ = viz.visualize_image_attr(attr_np, img_resized, method="blended_heat_map", sign="all", show_colorbar=True, alpha_overlay=0.6)
            fig.savefig(os.path.join(heatmaps_dir, f"{method_name}.png"), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            results.append({
                "Method": method_name, "Model": model_name, "Resolution": img_dims,
                "Input Size (px)": target_size,
                "Original Resolution": original_dims,
                "Prediction": predicted_class,
                "Device": device_info,
                "Model Cache": "reused" if was_model_cached else "loaded",
                "Timing Scope": "attribution_only",
                "Runtime (sec)": round(runtime_median, 4),
                "Attribution Runtime (sec)": round(runtime_median, 4),
                "Runtime Median (sec)": round(runtime_median, 4),
                "Attribution Runtime Median (sec)": round(runtime_median, 4),
                "Runtime Mean (sec)": round(runtime_mean, 4),
                "Attribution Runtime Mean (sec)": round(runtime_mean, 4),
                "Runtime Std (sec)": round(runtime_std, 4),
                "Attribution Runtime Std (sec)": round(runtime_std, 4),
                "Runtime Min (sec)": round(runtime_min, 4),
                "Attribution Runtime Min (sec)": round(runtime_min, 4),
                "Runtime Max (sec)": round(runtime_max, 4),
                "Attribution Runtime Max (sec)": round(runtime_max, 4),
                "Warmup Runs": warmup_runs,
                "Measured Runs": repeat_count,
                "Memory Scope": "attribution_peak",
                "Peak Memory (MB)": round(peak_memory_mb, 2),
                "Peak Attribution Memory (MB)": round(peak_memory_mb, 2)
            })

            # Explicitly delete objects and clear cache after each method
            del attribution, attr_np, xai_tool
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            gc.collect()

        except Exception as e:
            results.append({
                "Method": method_name, "Model": model_name, "Resolution": img_dims,
                "Original Resolution": original_dims,
                "Prediction": predicted_class if 'predicted_class' in locals() else "N/A",
                "Device": device_info,
                "Runtime (sec)": 0.0, "Peak Memory (MB)": 0.0, "Status": f"Error: {str(e)}"
            })

    # 6. Save Results
    pd.DataFrame(results).to_csv(os.path.join(session_dir, "results.csv"), index=False)
    task_completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(os.path.join(session_dir, "config.json"), 'w') as f:
        json.dump({
            **config,
            "prediction": predicted_class,
            "timing_scope": "attribution_only",
            "memory_scope": "attribution_peak",
            "task_started_at": task_started_at,
            "task_completed_at": task_completed_at,
            "environment": environment_metadata
        }, f, indent=4)

    # Final cleanup before returning to Streamlit
    del input_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    gc.collect()

    return results
