"""
Orchestrates the execution of XAI benchmark tasks.
Handles model inference, memory profiling, and timing measurements.
"""
import os
import sys
import re
import gc
import time
import json
import torch
import requests
import platform
import subprocess
import numpy as np
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt

try:
    import winreg
except ImportError:
    winreg = None

from PIL import Image
from io import BytesIO
from datetime import datetime
from dbgpu import GPUDatabase
from skimage.segmentation import slic
from memory_profiler import memory_usage
from backend.quality_runner import compute_quality_metrics

# Remove sys.path hack since models is now local to gui/

from torchvision import transforms
from utils.label_utils import get_label_mapping
from utils.model_loader import load_model, preprocess_image
from captum.attr import visualization as viz
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
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if val:
                    return val.strip()
            except Exception:
                pass
            return platform.processor()
        elif system == "Darwin":
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

def find_cpu_tdp(cpu_name):
    if not cpu_name or cpu_name.strip().lower() in ["", "generic cpu", "unknown"]:
        return None, None
        
    query = cpu_name.lower().strip()
    for term in ["(tm)", "(r)", "cpu", "@", "processor", "cores", "core", "graphics", "with", "\uFFFD", "®", "™"]:
        query = query.replace(term, " ")
    query = " ".join(query.split())
    
    is_intel = "intel" in query
    is_amd = "amd" in query or "ryzen" in query or "athlon" in query or "epyc" in query
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(os.path.dirname(base_dir), "assets")
    intel_path = os.path.join(assets_dir, "intel-cpus.csv")
    amd_path = os.path.join(assets_dir, "amd-cpus.csv")
    
    # Auto-download datasets if they don't exist locally
    if not os.path.exists(intel_path) or not os.path.exists(amd_path):
        try:
            os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)
            if not os.path.exists(intel_path):
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/felixsteinke/cpu-spec-dataset/main/dataset/intel-cpus.csv",
                    intel_path
                )
            if not os.path.exists(amd_path):
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/felixsteinke/cpu-spec-dataset/main/dataset/amd-cpus.csv",
                    amd_path
                )
        except Exception:
            pass
    
    matches = []
    
    # 1. Search Intel
    if (is_intel or not is_amd) and os.path.exists(intel_path):
        try:
            df = pd.read_csv(intel_path)
            for _, row in df.iterrows():
                pnum = str(row.get("ProcessorNumber", "")).lower().strip()
                cname = str(row.get("CpuName", "")).lower().strip()
                for term in ["\uFFFD", "®", "™"]:
                    cname = cname.replace(term, "")
                for term in ["intel", "processor", "graphics", "with", "(tm)", "(r)"]:
                    cname = cname.replace(term, " ")
                cname = " ".join(cname.split())
                
                if pnum and pnum in query:
                    matches.append((row.get("CpuName"), row.get("MaxTDP"), len(pnum)))
                elif cname and (cname in query or query in cname) and len(cname) > 4:
                    matches.append((row.get("CpuName"), row.get("MaxTDP"), len(cname)))
        except Exception:
            pass
            
    # 2. Search AMD
    if (is_amd or not is_intel) and os.path.exists(amd_path):
        try:
            df = pd.read_csv(amd_path)
            for _, row in df.iterrows():
                model = str(row.get("Model", "")).lower().strip()
                model_clean = model
                for term in ["\uFFFD", "®", "™"]:
                    model_clean = model_clean.replace(term, "")
                for term in ["amd", "ryzen", "athlon", "processor", "graphics", "with", "(tm)", "(r)"]:
                    model_clean = model_clean.replace(term, " ")
                model_clean = " ".join(model_clean.split())
                
                if model_clean and (model_clean in query or query in model_clean) and len(model_clean) > 3:
                    matches.append((row.get("Model"), row.get("Default TDP"), len(model_clean)))
        except Exception:
            pass
            
    if not matches:
        return None, None
        
    matches.sort(key=lambda x: x[2], reverse=True)
    best_match_name, tdp_str, _ = matches[0]
    
    tdp_val = None
    if tdp_str and pd.notna(tdp_str):
        m = re.search(r"(\d+)", str(tdp_str))
        if m:
            tdp_val = int(m.group(1))
            
    if best_match_name:
        best_match_name = best_match_name.replace("\uFFFD", "").replace("®", "").replace("™", "").strip()
        best_match_name = " ".join(best_match_name.split())
        
    return tdp_val, best_match_name

def collect_environment_metadata(device=None, custom_cpu_tdp=None, custom_gpu_tdp=None):
    cuda_devices = []
    if torch.cuda.is_available():
        db = None
        try:
            db = GPUDatabase.default()
        except Exception:
            pass

        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpu_name = torch.cuda.get_device_name(idx)
            
            tdp_val = None
            matched_gpu = None
            if db is not None:
                try:
                    spec = db.search(gpu_name)
                    if spec and hasattr(spec, "thermal_design_power_w") and spec.thermal_design_power_w:
                        tdp_val = int(spec.thermal_design_power_w)
                        matched_gpu = getattr(spec, "name", None)
                except Exception:
                    pass

            if custom_gpu_tdp is not None:
                if tdp_val is None:
                    matched_gpu = "User Specified"
                elif tdp_val != custom_gpu_tdp:
                    matched_gpu = "User Override"
                tdp_val = custom_gpu_tdp

            cuda_devices.append({
                "index": idx,
                "name": gpu_name,
                "total_memory_mb": round(props.total_memory / (1024 * 1024), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "tdp_w": tdp_val,
                "matched_name": matched_gpu,
            })

    if device == "mps" and custom_gpu_tdp is not None:
        if not cuda_devices:
            cuda_devices.append({
                "index": 0,
                "name": "Apple Silicon GPU (MPS)",
                "total_memory_mb": 0.0,
                "compute_capability": "N/A",
                "tdp_w": custom_gpu_tdp,
                "matched_name": "User Specified",
            })

    cpu_name = get_cpu_name()
    cpu_tdp = None
    matched_cpu = None
    try:
        cpu_tdp, matched_cpu = find_cpu_tdp(cpu_name)
    except Exception:
        pass

    if custom_cpu_tdp is not None:
        if cpu_tdp is None:
            matched_cpu = "User Specified"
        elif cpu_tdp != custom_cpu_tdp:
            matched_cpu = "User Override"
        cpu_tdp = custom_cpu_tdp

    return {
        "app_version": "1.0.0",
        "git_commit": get_git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": cpu_name,
        "cpu_tdp_w": cpu_tdp,
        "matched_cpu_name": matched_cpu,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available() or (device == "mps"),
        "cuda_device_count": len(cuda_devices),
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
    environment_metadata = collect_environment_metadata(
        device,
        custom_cpu_tdp=config.get('custom_cpu_tdp'),
        custom_gpu_tdp=config.get('custom_gpu_tdp')
    )

    # Determine active device TDP and convert to kW
    active_tdp_w = None
    if device.type in ['cuda', 'mps']:
        active_tdp_w = config.get('custom_gpu_tdp')
        if active_tdp_w is None and environment_metadata.get("cuda_devices"):
            active_tdp_w = environment_metadata["cuda_devices"][0].get("tdp_w")
    else:
        active_tdp_w = config.get('custom_cpu_tdp')
        if active_tdp_w is None:
            active_tdp_w = environment_metadata.get("cpu_tdp_w")
            
    active_tdp_kw = None
    if active_tdp_w is not None:
        try:
            active_tdp_kw = round(float(active_tdp_w) / 1000.0, 4)
        except Exception:
            pass

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

            def get_attr(inputs=None):
                inputs_to_use = input_tensor if inputs is None else inputs
                if method_key == 'integrated_gradients':
                    n_steps = int(method_params.get("n_steps", 50))
                    internal_batch_size = method_params.get("internal_batch_size", 2)
                    if internal_batch_size is not None:
                        internal_batch_size = int(internal_batch_size)
                    
                    base_mode = method_params.get("baseline_mode", "Zeros (Black)")
                    if base_mode == "Zeros (Black)":
                        baselines = torch.zeros_like(inputs_to_use)
                    elif base_mode == "Ones (White)":
                        baselines = torch.ones_like(inputs_to_use)
                    elif base_mode == "Input Mean":
                        baselines = torch.ones_like(inputs_to_use) * inputs_to_use.mean()
                    else:
                        baselines = torch.zeros_like(inputs_to_use)
                        
                    return xai_tool.attribute(inputs_to_use, target=pred_label_idx, n_steps=n_steps, internal_batch_size=internal_batch_size, baselines=baselines)
                if method_key == 'gradient_shap':
                    n_samples = int(method_params.get("n_samples", 10))
                    stdevs = float(method_params.get("stdevs", 0.0001))
                    
                    base_mode = method_params.get("baseline_mode", "Zeros & Mean")
                    if base_mode == "Zeros & Mean":
                        baseline_dist = torch.cat([torch.zeros_like(inputs_to_use), torch.ones_like(inputs_to_use) * inputs_to_use.mean()], dim=0)
                    elif base_mode == "Zeros Only":
                        baseline_dist = torch.zeros_like(inputs_to_use)
                    elif base_mode == "Ones Only":
                        baseline_dist = torch.ones_like(inputs_to_use)
                    else:
                        baseline_dist = torch.cat([torch.zeros_like(inputs_to_use), torch.ones_like(inputs_to_use) * inputs_to_use.mean()], dim=0)
                        
                    return xai_tool.attribute(inputs_to_use, baselines=baseline_dist, target=pred_label_idx, n_samples=n_samples, stdevs=stdevs)
                if method_key == 'deeplift':
                    return xai_tool.attribute(inputs_to_use, baselines=torch.zeros_like(inputs_to_use), target=pred_label_idx)
                if method_key == 'deeplift_shap':
                    baseline_dist = torch.cat([torch.zeros_like(inputs_to_use), torch.ones_like(inputs_to_use) * inputs_to_use.mean()], dim=0)
                    return xai_tool.attribute(inputs_to_use, baselines=baseline_dist, target=pred_label_idx)
                if method_key == 'grad_cam':
                    attribution = xai_tool.attribute(inputs_to_use, target=pred_label_idx)
                    attribution = LayerAttribution.interpolate(attribution, inputs_to_use.shape[2:])
                    return attribution.repeat(1, 3, 1, 1)
                if method_key == 'occlusion':
                    w_shapes = method_params.get("sliding_window_shapes", (3, 15, 15))
                    strds = method_params.get("strides", (3, 8, 8))
                    
                    if isinstance(w_shapes, list):
                        w_shapes = tuple(w_shapes)
                    if isinstance(strds, list):
                        strds = tuple(strds)
                    
                    occ_color = method_params.get("occlude_color", "0")
                    if occ_color == "mean":
                        baselines = inputs_to_use.mean().item()
                    else:
                        try:
                            baselines = float(occ_color)
                        except Exception:
                            baselines = 0.0
                            
                    return xai_tool.attribute(inputs_to_use, sliding_window_shapes=w_shapes, strides=strds, target=pred_label_idx, baselines=baselines)
                if method_key == 'lime':
                    n_samples = int(method_params.get("n_samples", 500))
                    batch_size = int(method_params.get("perturbations_per_eval", 10))
                    n_segments = int(method_params.get("n_segments", 50))
                    
                    img_np = inputs_to_use.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    superpixels = slic(img_np, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
                    superpixels = superpixels - superpixels.min()
                    feature_mask = torch.tensor(superpixels, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
                    
                    return xai_tool.attribute(inputs_to_use, target=pred_label_idx, feature_mask=feature_mask, n_samples=n_samples, perturbations_per_eval=batch_size)
                if method_key == 'saliency':
                    return xai_tool.attribute(inputs_to_use, target=pred_label_idx, abs=False)
                return xai_tool.attribute(inputs_to_use, target=pred_label_idx)

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
                    selected_metrics=selected_quality_metrics,
                    explanation_func=get_attr
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
            
            fig, _ = viz.visualize_image_attr(attr_np, img_resized, method="blended_heat_map", sign="all", show_colorbar=True, alpha_overlay=0.6, use_pyplot=False)
            fig.savefig(os.path.join(heatmaps_dir, f"{method_name}.png"), bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            # Calculate estimated energy consumption in kW
            est_energy = None
            if active_tdp_w is not None:
                try:
                    est_energy = round((float(runtime_median) * float(active_tdp_w)) / (3600.0 * 1000.0), 8)
                except Exception:
                    pass

            results.append({
                "Method": method_name, "Model": model_name, "Resolution": img_dims,
                "Input Size (px)": target_size,
                "Original Resolution": original_dims,
                "Prediction": predicted_class,
                "Device": device_info,
                "Estimated Energy Consumption (kW)": est_energy,
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
                "Sensitivity (Max)": round(quality_scores["Sensitivity (Max)"], 4) if quality_scores.get("Sensitivity (Max)") is not None else None,
                "Infidelity": round(quality_scores["Infidelity"], 4) if quality_scores.get("Infidelity") is not None else None,
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
                "Estimated Energy Consumption (kW)": None,
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
                "Sensitivity (Max)": None,
                "Infidelity": None,
                "Status": f"Failed: {str(e)}"
            })

    # 6. Save Results
    os.makedirs(session_dir, exist_ok=True)
    csv_path = os.path.join(session_dir, "results.csv")
    new_df = pd.DataFrame(results)
    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            # Combine and keep the latest run for each method
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df["_method_lower"] = combined_df["Method"].str.lower()
            combined_df = combined_df.drop_duplicates(subset=["_method_lower"], keep="last").drop(columns=["_method_lower"])
            combined_df.to_csv(csv_path, index=False)
        except Exception:
            new_df.to_csv(csv_path, index=False)
    else:
        new_df.to_csv(csv_path, index=False)
    task_completed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with open(os.path.join(session_dir, "config.json"), 'w') as f:
        json.dump({
            **config,
            "prediction": predicted_class,
            "original_resolution": original_dims,
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
