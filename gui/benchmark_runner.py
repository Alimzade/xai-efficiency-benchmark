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

from models.model_loader import load_model, preprocess_image, FIXED_SIZE_MODELS
from torchvision import transforms
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
    Occlusion,
    Lime,
)
from captum.attr import visualization as viz

try:
    from gui.quality_runner import compute_quality_metrics
except ImportError:
    from quality_runner import compute_quality_metrics

MODEL_CACHE = {}
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ViTReshapeWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = None
    def forward(self, x):
        self.cls_token = x[:, :1, :]
        patches = x[:, 1:, :]
        B, N, C = patches.shape
        grid_size = int(N ** 0.5)
        return patches.transpose(1, 2).reshape(B, C, grid_size, grid_size)

class ViTInverseReshapeWrapper(torch.nn.Module):
    def __init__(self, reshape_wrapper):
        super().__init__()
        self.reshape_wrapper = reshape_wrapper
    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.reshape(B, C, H * W).transpose(1, 2)
        return torch.cat([self.reshape_wrapper.cls_token, patches], dim=1)

GRAD_CAM_TARGET_LAYERS = {
    "resnet50": lambda model: model.layer4[-1],
    "convnext-t": lambda model: model.features[-1],
    "efficientnet-b0": lambda model: model.features[-1],
    "swin-t": lambda model: model.permute,
    "regnet-y-8gf": lambda model: model.trunk_output.block4,
    "mobilenet-v3-large": lambda model: model.features[-1],
    "densenet121": lambda model: model.features.denseblock4,
    "vit-b-16": lambda model: model.encoder.layers[-1].ln_1,
}

def is_mps_available():
    return hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

def sync_device(device):
    """Wait for queued CUDA/MPS work so wall-clock timing reflects actual GPU work."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        if is_mps_available() and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()

def get_cached_model(model_name, device):
    cache_key = (model_name, str(device))
    
    # Keep at most 2 different models in cache to prevent VRAM accumulation OOM
    if cache_key not in MODEL_CACHE and len(MODEL_CACHE) >= 2:
        oldest_key = list(MODEL_CACHE.keys())[0]
        del MODEL_CACHE[oldest_key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif is_mps_available() and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        gc.collect()

    was_cached = cache_key in MODEL_CACHE
    if not was_cached:
        MODEL_CACHE[cache_key] = load_model(model_name=model_name, device=device)
    return MODEL_CACHE[cache_key], was_cached

def get_grad_cam_target_layer(model_name, model):
    if model_name not in GRAD_CAM_TARGET_LAYERS:
        raise ValueError(f"Grad_CAM is not configured for model '{model_name}'.")
    
    if model_name == "vit-b-16":
        # Check if already wrapped
        already_wrapped = False
        reshape_layer = None
        target_block = model.encoder.layers[-1]
        if isinstance(target_block.ln_1, torch.nn.Sequential):
            for m in target_block.ln_1:
                if m.__class__.__name__ == "ViTReshapeWrapper":
                    already_wrapped = True
                    reshape_layer = m
                    break
        
        if not already_wrapped:
            orig_ln1 = target_block.ln_1
            reshape_layer = ViTReshapeWrapper()
            inv_reshape_layer = ViTInverseReshapeWrapper(reshape_layer)
            target_block.ln_1 = torch.nn.Sequential(orig_ln1, reshape_layer, inv_reshape_layer)
            
        return reshape_layer

    return GRAD_CAM_TARGET_LAYERS[model_name](model)

def normalize_method_name(method_name):
    # Strip any parameters in parentheses
    name = method_name.split("(")[0].strip()
    # Strip trailing numeric suffixes (e.g. Lime_1 -> Lime)
    import re
    name = re.sub(r'_\d+$', '', name)
    return name.lower().replace("-", "_")

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

def get_cpu_name():
    try:
        system = platform.system()
        if system == "Windows":
            return platform.processor()
        elif system == "Darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        elif system == "Linux":
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
                # Fallback for ARM Linux (like Raspberry Pi)
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "Hardware" in line or "Processor" in line:
                            return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Generic CPU"

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
        "processor": get_cpu_name(),
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
    import re
    norm_model_name = re.sub(r'_\d+$', '', model_name)
    model, was_model_cached = get_cached_model(model_name=norm_model_name, device=device)

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
    input_tensor = preprocess_image(img, model_name=norm_model_name, target_size=target_size).unsqueeze(0).to(device)
    img_dims = f"{input_tensor.shape[2]} x {input_tensor.shape[3]}"

    # 4. Get Prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, pred_label_idx = torch.max(output, 1)
        predicted_class, _ = get_label_mapping(
            model_name=norm_model_name, predicted_class=pred_label_idx, label=None, label_names=None
        )

    # 5. Benchmarking Loop
    results = []
    methods_to_run = config.get('methods', ['saliency'])
    warmup_runs = max(0, int(config.get('warmup_runs', 1)))
    memory_runs = max(0, int(config.get('memory_runs', 1)))
    repeat_count = max(1, int(config.get('repeat_count', 1)))
    enable_quality_metrics = config.get('enable_quality_metrics', False)
    selected_quality_metrics = config.get('selected_quality_metrics', ["Gini Index (Sparsity)"]) if enable_quality_metrics else []
    heatmaps_dir = os.path.join(session_dir, "heatmaps")
    os.makedirs(heatmaps_dir, exist_ok=True)

    for method_name in methods_to_run:
        try:
            method_key = normalize_method_name(method_name)
            # Clear cache before every method
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            method_params = config.get("method_params", {})

            if method_key == 'saliency': xai_tool = Saliency(model)
            elif method_key == 'integrated_gradients': xai_tool = IntegratedGradients(model)
            elif method_key == 'guided_backprop': xai_tool = GuidedBackprop(model)
            elif method_key == 'input_x_gradient': xai_tool = InputXGradient(model)
            elif method_key == 'gradient_shap': xai_tool = GradientShap(model)
            elif method_key == 'deeplift': xai_tool = DeepLift(model)
            elif method_key == 'deeplift_shap': xai_tool = DeepLiftShap(model)
            elif method_key == 'grad_cam': xai_tool = LayerGradCam(model, get_grad_cam_target_layer(norm_model_name, model))
            elif method_key == 'occlusion': xai_tool = Occlusion(model)
            elif method_key == 'lime': xai_tool = Lime(model)
            else: continue

            def get_attr():
                if method_key == 'integrated_gradients':
                    n_steps = int(method_params.get("n_steps", 50))
                    internal_batch_size = method_params.get("internal_batch_size", 2)
                    if internal_batch_size is not None:
                        internal_batch_size = int(internal_batch_size)
                    
                    base_mode = method_params.get("baseline_mode", "Zeros (Black)")
                    if base_mode == "Zeros (Black)":
                        baselines = torch.zeros_like(input_tensor)
                    elif base_mode == "Ones (White)":
                        baselines = torch.ones_like(input_tensor)
                    elif base_mode == "Input Mean":
                        baselines = torch.ones_like(input_tensor) * input_tensor.mean()
                    else:
                        baselines = torch.zeros_like(input_tensor)
                        
                    return xai_tool.attribute(input_tensor, target=pred_label_idx, n_steps=n_steps, internal_batch_size=internal_batch_size, baselines=baselines)
                if method_key == 'gradient_shap':
                    n_samples = int(method_params.get("n_samples", 10))
                    stdevs = float(method_params.get("stdevs", 0.0001))
                    
                    base_mode = method_params.get("baseline_mode", "Zeros & Mean")
                    if base_mode == "Zeros & Mean":
                        baseline_dist = torch.cat([torch.zeros_like(input_tensor), torch.ones_like(input_tensor) * input_tensor.mean()], dim=0)
                    elif base_mode == "Zeros Only":
                        baseline_dist = torch.zeros_like(input_tensor)
                    elif base_mode == "Ones Only":
                        baseline_dist = torch.ones_like(input_tensor)
                    else:
                        baseline_dist = torch.cat([torch.zeros_like(input_tensor), torch.ones_like(input_tensor) * input_tensor.mean()], dim=0)
                        
                    return xai_tool.attribute(input_tensor, baselines=baseline_dist, target=pred_label_idx, n_samples=n_samples, stdevs=stdevs)
                if method_key == 'deeplift':
                    return xai_tool.attribute(input_tensor, baselines=torch.zeros_like(input_tensor), target=pred_label_idx)
                if method_key == 'deeplift_shap':
                    baseline_dist = torch.cat([torch.zeros_like(input_tensor), torch.ones_like(input_tensor) * input_tensor.mean()], dim=0)
                    return xai_tool.attribute(input_tensor, baselines=baseline_dist, target=pred_label_idx)
                if method_key == 'grad_cam':
                    attribution = xai_tool.attribute(input_tensor, target=pred_label_idx)
                    attribution = LayerAttribution.interpolate(attribution, input_tensor.shape[2:])
                    return attribution.repeat(1, 3, 1, 1)
                if method_key == 'occlusion':
                    w_shapes = method_params.get("sliding_window_shapes", (3, 15, 15))
                    strds = method_params.get("strides", (3, 8, 8))
                    
                    occ_color = method_params.get("occlude_color", "0")
                    if occ_color == "mean":
                        baselines = input_tensor.mean().item()
                    else:
                        try:
                            baselines = float(occ_color)
                        except Exception:
                            baselines = 0.0
                            
                    return xai_tool.attribute(input_tensor, sliding_window_shapes=w_shapes, strides=strds, target=pred_label_idx, baselines=baselines)
                if method_key == 'lime':
                    from skimage.segmentation import slic
                    n_samples = int(method_params.get("n_samples", 500))
                    batch_size = int(method_params.get("perturbations_per_eval", 10))
                    n_segments = int(method_params.get("n_segments", 50))
                    
                    img_np = input_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    superpixels = slic(img_np, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
                    superpixels = superpixels - superpixels.min()
                    feature_mask = torch.tensor(superpixels, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
                    
                    return xai_tool.attribute(input_tensor, target=pred_label_idx, feature_mask=feature_mask, n_samples=n_samples, perturbations_per_eval=batch_size)
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
                        if is_mps_available() and hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        gc.collect()
                        memory_before = torch.mps.current_allocated_memory() if (is_mps_available() and hasattr(torch.mps, 'current_allocated_memory')) else 0

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
                        memory_after = torch.mps.current_allocated_memory() if (is_mps_available() and hasattr(torch.mps, 'current_allocated_memory')) else 0
                        peak_memory_mb = max(memory_after - memory_before, 0) / (1024 * 1024)

                return attribution_result, runtime_sec, peak_memory_mb

            for _ in range(warmup_runs):
                warmup_attribution, _, _ = timed_get_attr(measure_memory=False)
                del warmup_attribution
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif is_mps_available() and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                gc.collect()

            # --- DEDICATED MEMORY MEASUREMENT (After warmups) ---
            memory_values = []
            for _ in range(memory_runs):
                if device.type in ['cuda', 'mps']:
                    # On GPU, memory can be measured in a clean run with zero overhead
                    _, _, current_memory = timed_get_attr(measure_memory=True)
                else:
                    # On CPU, run the memory profiler to measure peak memory without contaminating timing repeats
                    mem_usage, _ = memory_usage((timed_get_attr, (False,)), interval=0.1, retval=True)
                    current_memory = max(mem_usage) - min(mem_usage) if mem_usage else 0.0
                memory_values.append(current_memory if current_memory is not None else 0.0)

            peak_memory_mb = float(np.mean(memory_values)) if memory_values else None
            memory_std = float(np.std(memory_values)) if len(memory_values) > 1 else None

            # --- CLEAN TIMING REPEATS ---
            attribution = None
            runtime_values = []

            for _ in range(repeat_count):
                # Clean timing run (no background memory profiling thread)
                current_attribution, current_runtime, _ = timed_get_attr(measure_memory=False)

                if attribution is not None:
                    del attribution
                attribution = current_attribution
                runtime_values.append(current_runtime)

            runtime_median = float(np.median(runtime_values))
            runtime_mean = float(np.mean(runtime_values))
            runtime_std = float(np.std(runtime_values)) if len(runtime_values) > 1 else None
            runtime_min = float(np.min(runtime_values))
            runtime_max = float(np.max(runtime_values))
            
            # --- PHASE B: EXPLANATION QUALITY METRICS (OFF TIMING CLOCK) ---
            quality_scores = {}
            if enable_quality_metrics and selected_quality_metrics:
                quality_scores = compute_quality_metrics(
                    attribution=attribution,
                    model=model,
                    input_tensor=input_tensor,
                    target_class=pred_label_idx,
                    method_key=method_key,
                    device=device,
                    selected_metrics=selected_quality_metrics
                )

            # Generate Overlay — background must match the exact Resize+CenterCrop the model saw
            attr_np = np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))
            
            # Apply the exact spatial preprocessing as preprocess_image (Resize → CenterCrop)
            # so the background image is pixel-aligned with the attribution tensor
            resize_val = int(target_size * (256 / 224))
            crop_transform = transforms.Compose([
                transforms.Resize(resize_val),
                transforms.CenterCrop(target_size),
            ])
            img_cropped = crop_transform(img)
            img_resized = np.array(img_cropped)
            
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
                "Runtime Std (sec)": round(runtime_std, 4) if runtime_std is not None else None,
                "Attribution Runtime Std (sec)": round(runtime_std, 4) if runtime_std is not None else None,
                "Runtime Min (sec)": round(runtime_min, 4),
                "Attribution Runtime Min (sec)": round(runtime_min, 4),
                "Runtime Max (sec)": round(runtime_max, 4),
                "Attribution Runtime Max (sec)": round(runtime_max, 4),
                "Warmup Runs": warmup_runs,
                "Memory Runs": memory_runs,
                "Measured Runs": repeat_count,
                "Memory Scope": "attribution_peak",
                "Peak Memory (MB)": round(peak_memory_mb, 2) if peak_memory_mb is not None else None,
                "Peak Attribution Memory (MB)": round(peak_memory_mb, 2) if peak_memory_mb is not None else None,
                "Attribution Memory Std (MB)": round(memory_std, 2) if memory_std is not None else None,
                "Gini Index": round(quality_scores["Gini Index"], 4) if quality_scores.get("Gini Index") is not None else None,
                "Deletion AUC": round(quality_scores["Deletion AUC"], 4) if quality_scores.get("Deletion AUC") is not None else None,
                "Insertion AUC": round(quality_scores["Insertion AUC"], 4) if quality_scores.get("Insertion AUC") is not None else None,
                "Infidelity": round(quality_scores["Infidelity"], 4) if quality_scores.get("Infidelity") is not None else None,
                "Quality Eval Time (sec)": round(quality_scores["Quality Eval Time (sec)"], 4) if quality_scores.get("Quality Eval Time (sec)") is not None else None
            })

            # Explicitly delete objects and clear cache after each method
            del attribution, attr_np, xai_tool
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif is_mps_available() and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            gc.collect()

        except Exception as e:
            results.append({
                "Method": method_name, "Model": model_name, "Resolution": img_dims if 'img_dims' in locals() else f"{target_size} x {target_size}",
                "Input Size (px)": target_size,
                "Original Resolution": original_dims,
                "Prediction": predicted_class if 'predicted_class' in locals() else "N/A",
                "Device": device_info,
                "Model Cache": "reused" if was_model_cached else "loaded",
                "Timing Scope": "attribution_only",
                "Runtime (sec)": 0.0, "Attribution Runtime (sec)": 0.0,
                "Runtime Median (sec)": 0.0, "Attribution Runtime Median (sec)": 0.0,
                "Runtime Mean (sec)": 0.0, "Attribution Runtime Mean (sec)": 0.0,
                "Runtime Std (sec)": 0.0, "Attribution Runtime Std (sec)": 0.0,
                "Runtime Min (sec)": 0.0, "Attribution Runtime Min (sec)": 0.0,
                "Runtime Max (sec)": 0.0, "Attribution Runtime Max (sec)": 0.0,
                "Warmup Runs": warmup_runs,
                "Memory Runs": memory_runs,
                "Measured Runs": repeat_count,
                "Memory Scope": "attribution_peak",
                "Peak Memory (MB)": None, "Peak Attribution Memory (MB)": None,
                "Attribution Memory Std (MB)": None,
                "Gini Index": None,
                "Deletion AUC": None,
                "Insertion AUC": None,
                "Infidelity": None,
                "Quality Eval Time (sec)": None,
                "Status": f"Failed: {str(e)}"
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
    elif is_mps_available() and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    gc.collect()

    return results
