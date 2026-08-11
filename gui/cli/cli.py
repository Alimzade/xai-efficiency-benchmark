#!/usr/bin/env python3
import os
import sys

# Setup paths so imports from 'gui' (config, backend, utils, analysis, components) work regardless of working directory
CLI_DIR = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.abspath(os.path.join(CLI_DIR, ".."))
if GUI_DIR not in sys.path:
    sys.path.insert(0, GUI_DIR)

import json
import time
import torch
import platform
import argparse
import warnings
import pandas as pd
from datetime import datetime

from analysis.metrics import method_detail_summary, image_size_summary
from analysis.pareto import compute_pareto_ranking
from utils.processing import normalize_metric_columns, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL, ATTR_MEMORY_COL, LEGACY_MEMORY_COL, metric_col

# Filter benign framework UserWarnings (e.g. Captum required_grads tensor notices)
warnings.filterwarnings("ignore", category=UserWarning)

def resolve_path(p):
    """Resolve a path whether executed from repo root, gui/, or cli/ directory."""
    if not p:
        return p
    p = p.strip().strip(' "\'')
    if os.path.exists(p):
        return p
    # Check relative to GUI_DIR
    p_gui = os.path.join(GUI_DIR, p)
    if os.path.exists(p_gui):
        return p_gui
    # Strip leading 'gui/' or 'gui\' if entered from inside the gui directory
    if p.startswith("gui/") or p.startswith("gui\\"):
        stripped = p[4:]
        if os.path.exists(stripped):
            return stripped
        if os.path.exists(os.path.join(GUI_DIR, stripped)):
            return os.path.join(GUI_DIR, stripped)
    return p

def prompt_for_images_wizard(questionary):
    """Interactively prompt for images, verifying count and looping back if no images are found."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    while True:
        input_type = questionary.select(
            "How would you like to provide images?",
            choices=[
                "Scan a directory for images",
                "Provide a single local file path",
                "Provide a single URL",
                "Provide a .txt file containing multiple paths/URLs"
            ]
        ).ask()
        if not input_type:
            sys.exit(0)

        if "directory" in input_type:
            default_dir = "images/" if os.path.isdir(os.path.join(GUI_DIR, "images")) else ""
            images_input = questionary.text(
                "Enter the directory path (e.g. images/):",
                default=default_dir,
                validate=lambda x: os.path.isdir(resolve_path(x)) or "Please enter a valid directory path."
            ).ask()
            if images_input is None:
                sys.exit(0)
            resolved = resolve_path(images_input)
            
            found_count = 0
            for root, _, files in os.walk(resolved):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        found_count += 1
            if found_count > 0:
                print(f"  -> Detected {found_count} image(s) in '{images_input}'\n")
                return resolved
            else:
                print(f"\n[!] No valid image files (.jpg, .png, .webp, .bmp) found in '{images_input}'. Please choose a different source.\n")
                continue

        elif "local file" in input_type:
            images_input = questionary.text(
                "Enter the local file path (e.g. images/my_image.jpg):",
                validate=lambda x: os.path.isfile(resolve_path(x)) or "Please enter a valid file path."
            ).ask()
            if images_input is None:
                sys.exit(0)
            resolved = resolve_path(images_input)
            ext = os.path.splitext(resolved)[1].lower()
            if ext in valid_exts:
                print(f"  -> Loaded 1 local image: '{images_input}'\n")
                return resolved
            else:
                print(f"\n[!] File '{images_input}' is not a supported image format ({', '.join(valid_exts)}).\n")
                continue

        elif "URL" in input_type:
            images_input = questionary.text(
                "Enter the URL:",
                validate=lambda x: x.startswith("http://") or x.startswith("https://") or "URL must start with http:// or https://"
            ).ask()
            if images_input is None:
                sys.exit(0)
            print(f"  -> Loaded remote image: '{images_input}'\n")
            return images_input

        elif ".txt" in input_type:
            images_input = questionary.text(
                "Enter the path to your .txt file (e.g. image_urls.txt):",
                validate=lambda x: (x.endswith(".txt") and os.path.isfile(resolve_path(x))) or "Please enter a valid .txt file path."
            ).ask()
            if images_input is None:
                sys.exit(0)
            resolved = resolve_path(images_input)
            with open(resolved, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                print(f"  -> Loaded {len(lines)} image source(s) from '{images_input}'\n")
                return resolved
            else:
                print(f"\n[!] The text file '{images_input}' is empty. Please choose a different source.\n")
                continue

# Heavy imports are deferred to main() to allow --help to run instantly



def run_interactive_wizard():
    try:
        import questionary
    except ImportError:
        print("\n[!] The interactive wizard requires the 'questionary' library.")
        print("[!] Please run: pip install questionary")
        return False

    try:
        from config import model_opts, xai_opts, fixed_size_models, min_input_size, default_input_size, region_based_methods
    except Exception as e:
        print(f"\n[!] Backend Error: {e}")
        return False

    print("\n" + "="*60)
    print("🚀 XAI EFFICIENCY BENCHMARK INTERACTIVE SETUP 🚀")
    print("="*60 + "\n")
    
    last_run_file = os.path.join(CLI_DIR, "assets", ".last_run_config.json")
    if os.path.exists(last_run_file):
        restore = questionary.select(
            "Found a previous execution configuration. What would you like to do?",
            choices=["Run previous configuration exactly as is", "Run previous configuration, but change images", "Start fresh"]
        ).ask()
        if restore is None: sys.exit(0)
        
        if "exactly as is" in restore:
            sys.argv.extend(["--config", last_run_file])
            return True
        elif "change images" in restore:
            try:
                with open(last_run_file, "r") as f:
                    cfg = json.load(f)
            except Exception as e:
                print(f"[!] Could not load previous config: {e}")
                sys.exit(1)
                
            images_input = prompt_for_images_wizard(questionary)
            
            cfg["images"] = images_input
            with open(last_run_file, "w") as f:
                json.dump(cfg, f)
            
            sys.argv.extend(["--config", last_run_file])
            return True
        
        print("")

    selected_models = questionary.checkbox(
        "Which models would you like to benchmark?",
        choices=model_opts,
        validate=lambda x: len(x) > 0 or "You must select at least one model."
    ).ask()
    if not selected_models: sys.exit(0)
    print(f"  -> {', '.join(selected_models)}\n")

    # Interactive Method & Parameter Setup
    tasks = []
    
    # 1. Fast Bulk Selection
    selected_methods = questionary.checkbox(
        "Which XAI methods would you like to run?",
        choices=xai_opts,
        validate=lambda x: len(x) > 0 or "You must select at least one method."
    ).ask()
    if not selected_methods: sys.exit(0)
    
    print(f"  -> {', '.join(selected_methods)}\n")
    
    # 2. Iterate through bulk selection for parameters
    for method in selected_methods:
        method_count = 1
        
        while True:
            save_name = method if method_count == 1 else f"{method}_{method_count}"
            params = {}
            from config import method_configs
            if method in method_configs:
                config = method_configs[method]
                for param_key, param_info in config.items():
                    prompt_label = f"[{save_name}] {param_info['label']}"
                    if param_info["type"] == "select":
                        opts = param_info["choices"]
                        default_str = str(param_info["default"])
                        
                        display_opts = []
                        for o in opts:
                            if o == default_str:
                                display_opts.append(f"{o} (Default)")
                            else:
                                display_opts.append(o)
                                
                        ans_opt = questionary.select(f"{prompt_label}:", choices=display_opts).ask()
                        if ans_opt is None: sys.exit(0)
                        
                        if ans_opt.endswith(" (Default)"):
                            params[param_key] = ans_opt[:-10]
                        else:
                            params[param_key] = ans_opt
                    else:
                        opts = param_info.get("choices", [str(param_info["default"]), "Custom"])
                        default_str = str(param_info["default"])
                        
                        display_opts = []
                        for o in opts:
                            if o == default_str:
                                display_opts.append(f"{o} (Default)")
                            else:
                                display_opts.append(o)
                                
                        ans_opt = questionary.select(f"{prompt_label}:", choices=display_opts).ask()
                        if ans_opt is None: sys.exit(0)
                        
                        if ans_opt == "Custom":
                            ans_val = questionary.text(f"Enter custom {prompt_label}:").ask()
                            if ans_val is None: sys.exit(0)
                            val_str = ans_val
                        else:
                            val_str = ans_opt.split()[0]
                            
                        if param_info["type"] == "int":
                            params[param_key] = int(val_str)
                        elif param_info["type"] == "float":
                            params[param_key] = float(val_str)
                        else:
                            params[param_key] = val_str
            
            tasks.append({"method": save_name, "params": params})
            
            if method in method_configs:
                print("")
                add_another = questionary.select(
                    f"Would you like to add another instance of {method} with different parameters?",
                    choices=["No", "Yes"]
                ).ask()
                if add_another is None: sys.exit(0)
                if add_another == "Yes":
                    method_count += 1
                else:
                    break
            else:
                break
    
    images_input = prompt_for_images_wizard(questionary)

    fixed_models = [m for m in selected_models if m in fixed_size_models]
    if fixed_models:
        formatted = ", ".join(fixed_models)
        print(f"[*] Fixed-size architecture selected ({formatted}). Input resolution locked to 224px.")
        sizes_input = "224"
    else:
        sizes_input = questionary.text(
            f"Input sizes in pixels [min: {min_input_size}px] (comma-separated):",
            default=str(default_input_size),
            validate=lambda x: (
                all(s.strip().isdigit() and int(s.strip()) >= min_input_size for s in x.split(",") if s.strip()) and len([s for s in x.split(",") if s.strip()]) > 0
            ) or f"Please enter valid pixel sizes (comma-separated integers >= {min_input_size}, e.g. 112, 224, 448)."
        ).ask()
        if sizes_input is None: sys.exit(0)
    
    print("\n[*] Detecting hardware...\n")
    from backend.benchmark_runner import collect_environment_metadata
    env_meta = collect_environment_metadata()
    
    print("="*40)
    print("💻 HARDWARE DETECTED:")
    print(f"  CPU: {env_meta.get('processor', 'Unknown')} (TDP: {env_meta.get('cpu_tdp_w', 'Unknown')}W)")
    gpu_list = env_meta.get('cuda_devices', [])
    gpu_tdp_opt = ""
    if gpu_list:
        gpu = gpu_list[0]
        print(f"  GPU: {gpu.get('name', 'Unknown')} (TDP: {gpu.get('tdp_w', 'Unknown')}W)")
    print("="*40 + "\n")
    
    import torch
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    device_opt = "cpu"
    if has_cuda:
        choice = questionary.select(
            "Execution device:",
            choices=["cuda (GPU - NVIDIA)", "cpu (CPU)"]
        ).ask()
        if choice is None: sys.exit(0)
        device_opt = "cuda" if "cuda" in choice else "cpu"
    elif has_mps:
        choice = questionary.select(
            "Execution device:",
            choices=["mps (Apple Silicon GPU)", "cpu (CPU)"]
        ).ask()
        if choice is None: sys.exit(0)
        device_opt = "mps" if "mps" in choice else "cpu"
    else:
        print("[*] Device: Locked to CPU (no compatible GPU detected).\n")
        device_opt = "cpu"
    
    cpu_tdp_opt = ""
    gpu_tdp_opt = ""
    
    if device_opt in ["cuda", "mps"]:
        det_gpu_tdp = gpu_list[0].get('tdp_w') if gpu_list else None
        if det_gpu_tdp:
            gpu_prompt = f"Override GPU TDP (Watts)? [Press Enter for detected {det_gpu_tdp}W]:"
            gpu_default = str(det_gpu_tdp)
        else:
            gpu_prompt = "GPU TDP not detected. Enter GPU TDP in Watts [default: 75W]:"
            gpu_default = "75"
            
        gpu_tdp_opt = questionary.text(
            gpu_prompt,
            default=gpu_default,
            validate=lambda x: (x.replace('.', '', 1).isdigit() and float(x) > 0) or "Enter a valid positive number."
        ).ask()
        if gpu_tdp_opt is None: sys.exit(0)
    else:
        det_cpu_tdp = env_meta.get('cpu_tdp_w')
        if det_cpu_tdp:
            cpu_prompt = f"Override CPU TDP (Watts)? [Press Enter for detected {det_cpu_tdp}W]:"
            cpu_default = str(det_cpu_tdp)
        else:
            cpu_prompt = "CPU TDP not detected. Enter CPU TDP in Watts [default: 45W]:"
            cpu_default = "45"
            
        cpu_tdp_opt = questionary.text(
            cpu_prompt,
            default=cpu_default,
            validate=lambda x: (x.replace('.', '', 1).isdigit() and float(x) > 0) or "Enter a valid positive number."
        ).ask()
        if cpu_tdp_opt is None: sys.exit(0)

    warmups_opt = questionary.text("Warmup iterations:", default="3", validate=lambda x: x.isdigit() or "Enter a number.").ask()
    if warmups_opt is None: sys.exit(0)
    
    repeats_opt = questionary.text("Measured repeats:", default="5", validate=lambda x: (x.isdigit() and int(x) > 0) or "Enter a positive number.").ask()
    if repeats_opt is None: sys.exit(0)
    
    memory_runs_opt = questionary.text("Memory runs:", default="1", validate=lambda x: x.isdigit() or "Enter a number.").ask()
    if memory_runs_opt is None: sys.exit(0)

    run_order_opt = questionary.select(
        "Task Execution Order:",
        choices=["Balanced", "Sequential", "Randomized"]
    ).ask()
    if run_order_opt is None: sys.exit(0)

    random_seed_opt = 42
    if run_order_opt == "Randomized":
        seed_ans = questionary.text(
            "Random seed for task shuffling:",
            default="42",
            validate=lambda x: x.isdigit() or "Please enter a valid integer seed."
        ).ask()
        if seed_ans is None: sys.exit(0)
        random_seed_opt = int(seed_ans)

    # Explanation Quality Metrics Selection
    eval_quality_ans = questionary.select(
        "Evaluate Explanation Quality Metrics (post-processing)?",
        choices=["No (Faster - Efficiency Only)", "Yes (Evaluate Faithfulness & Sparsity)"]
    ).ask()
    if eval_quality_ans is None: sys.exit(0)

    selected_quality_metrics = []
    if "Yes" in eval_quality_ans:
        while True:
            selected_quality_metrics = questionary.checkbox(
                "Select Quality Metrics to compute:",
                choices=[
                    "Gini Index (Sparsity)",
                    "Deletion AUC",
                    "Insertion AUC",
                    "Sensitivity (Max)",
                    "Infidelity (Perturbation Faithfulness)"
                ],
                validate=lambda x: len(x) > 0 or "Please select at least one metric (or choose 'No' to skip quality evaluation)."
            ).ask()
            if selected_quality_metrics is None: sys.exit(0)

            if any("Infidelity" in m for m in selected_quality_metrics):
                region_methods = [t.get("method") for t in tasks if any(rm in t.get("method", "") for rm in region_based_methods)]
                if region_methods:
                    print(f"\n[!] WARNING: Infidelity is mathematically ill-suited for region-based methods ({', '.join(region_methods)}).")
                    print("    Because coarse patch/superpixel attributions cannot track fine-grained pixel noise (delta^T A(x)),")
                    print("    Infidelity scores for these methods will be distorted and fundamentally invalid for direct comparison.\n")
                    
                    proceed_choice = questionary.select(
                        "How would you like to proceed?",
                        choices=[
                            "Exclude Infidelity and continue (Recommended)",
                            "Keep Infidelity anyway",
                            "Re-select Quality Metrics"
                        ]
                    ).ask()
                    if proceed_choice is None: sys.exit(0)
                    
                    if "Exclude Infidelity" in proceed_choice:
                        selected_quality_metrics = [m for m in selected_quality_metrics if "Infidelity" not in m]
                        print("  -> Excluded Infidelity. Proceeding with remaining quality metrics.\n")
                        break
                    elif "Re-select" in proceed_choice:
                        print("")
                        continue
                    else:
                        break
            break

    print("\n[*] Setup Complete! Initializing backend...\n")
    
    # Save JSON config to pass to CLI and for future restore
    config_file = os.path.join(CLI_DIR, "assets", ".last_run_config.json")
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    with open(config_file, "w") as f:
        json.dump({
            "models": selected_models,
            "tasks": tasks,
            "images": images_input,
            "input_sizes": sizes_input,
            "device": device_opt,
            "warmups": int(warmups_opt),
            "repeats": int(repeats_opt),
            "memory_runs": int(memory_runs_opt),
            "run_order": run_order_opt,
            "random_seed": int(random_seed_opt),
            "cpu_tdp": float(cpu_tdp_opt) if cpu_tdp_opt.strip() else None,
            "gpu_tdp": float(gpu_tdp_opt) if gpu_tdp_opt.strip() else None,
            "quality_metrics": ",".join(selected_quality_metrics)
        }, f)
    
    # Inject user selections into sys.argv so argparse parses them seamlessly
    sys.argv.extend([
        "--config", config_file
    ])
    return True

def get_or_create_cli_model_entry(global_results, img_i, src_img, model_name, size, sm, batch_id, args):
    img_id = f"Image {img_i+1}"
    img_group = next((g for g in global_results if g.get("image_id") == img_id or g.get("img_idx") == img_i), None)
    if not img_group:
        img_group = {
            "img_idx": img_i,
            "image_id": img_id,
            "source": src_img,
            "models": []
        }
        global_results.append(img_group)

    model_label = f"{model_name} ({size}px)"
    model_entry = next((m for m in img_group["models"] if m.get("model_label") == model_label), None)
    if not model_entry:
        s_dir = sm.get_task_path(batch_id, img_i + 1, f"{model_name}_{size}")
        os.makedirs(s_dir, exist_ok=True)
        dest_img = os.path.join(s_dir, "input_image.jpg")
        
        is_url = src_img.startswith("http://") or src_img.startswith("https://")
        if not os.path.exists(dest_img):
            if is_url:
                import urllib.request
                try:
                    urllib.request.urlretrieve(src_img, dest_img)
                except Exception as e:
                    print(f"[!] Warning: Failed to download {src_img}. Error: {e}")
            elif os.path.exists(src_img) and src_img != dest_img:
                import shutil
                shutil.copy(src_img, dest_img)

        config_path = os.path.join(s_dir, "config.json")
        if not os.path.exists(config_path):
            pred_config = {
                "model_name": model_name, 
                "image_source": dest_img, 
                "methods": [], 
                "method_params": {},
                "force_device": args.device if args.device != "auto" else None, 
                "input_size": size,
                "warmup_runs": 0,
                "memory_runs": 0,
                "repeat_count": 1,
                "run_order": args.run_order,
                "enable_quality_metrics": False,
                "selected_quality_metrics": [],
                "custom_cpu_tdp": args.cpu_tdp,
                "custom_gpu_tdp": args.gpu_tdp
            }
            print(f"[*] Running initial prediction pass for {model_name} ({size}px)...")
            from backend.benchmark_runner import run_benchmark_task
            run_benchmark_task(pred_config, s_dir)

        pred_val = "Unknown"
        orig_res_val = "Unknown"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    t_cfg = json.load(f)
                pred_val = t_cfg.get("prediction", "Unknown")
                orig_res_val = t_cfg.get("original_resolution", "Unknown")
            except Exception:
                pass

        model_entry = {
            "model": model_name,
            "model_label": model_label,
            "input_size": size,
            "session_dir": s_dir,
            "src_path": dest_img,
            "prediction": pred_val,
            "original_resolution": orig_res_val,
            "results": []
        }
        img_group["models"].append(model_entry)
        if pred_val != "Unknown":
            print(f"  -> Model Prediction: {pred_val}")

    return model_entry


def main():
    # Detect if launched via wrapper script or directly via python
    if os.environ.get("LAUNCHED_VIA_WRAPPER"):
        launcher_name = "Run_CLI.bat" if os.name == "nt" else "./run_cli.sh"
    else:
        launcher_name = os.path.basename(sys.argv[0])
    
    parser = argparse.ArgumentParser(
        prog=launcher_name,
        description="XAI Efficiency Benchmark CLI"
    )
    
    # Core Parameters
    parser.add_argument("--config", type=str, default="", help="Path to a JSON configuration file (bypasses other execution arguments)")
    parser.add_argument("--models", type=str, default="resnet50", help="Comma-separated list of models (e.g. resnet50,vit_b_16)")
    parser.add_argument("--methods", type=str, default="Saliency,Integrated_Gradients", help="Comma-separated list of XAI methods")
    parser.add_argument("--input-sizes", type=str, default="224", help="Comma-separated list of image sizes (e.g. 224,256)")
    parser.add_argument("--images", type=str, default="", help="Comma-separated files, directories, URLs, or .txt files")
    
    # Execution Parameters
    parser.add_argument("--warmups", type=int, default=3, help="Number of warmup iterations")
    parser.add_argument("--repeats", type=int, default=5, help="Number of measurement repeats")
    parser.add_argument("--memory-runs", type=int, default=1, help="Number of memory profiling runs")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to force execution on")
    parser.add_argument("--run-order", type=str, default="Balanced", choices=["Balanced", "Sequential", "Randomized"], help="Task execution order")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed used for task shuffling when --run-order is Randomized (default: 42).")
    
    # Hardware/Energy Parameters
    parser.add_argument("--cpu-tdp", type=float, default=None, help="Custom CPU TDP (W) for energy scaling")
    parser.add_argument("--gpu-tdp", type=float, default=None, help="Custom GPU TDP (W) for energy scaling")
    
    # Quality Parameters
    parser.add_argument("--quality-metrics", type=str, default="", help="Comma-separated quality metrics (e.g. 'Gini Index (Sparsity),Deletion AUC')")

    # Info Parameters
    parser.add_argument("--list-models", action="store_true", help="List all supported models and exit")
    parser.add_argument("--list-methods", action="store_true", help="List all supported XAI methods and exit")
    parser.add_argument("--list-metrics", action="store_true", help="List all supported quality metrics and exit")
    
    # If no arguments are provided, trigger the Interactive Wizard!
    if len(sys.argv) == 1:
        success = run_interactive_wizard()
        if not success:
            parser.print_help(sys.stderr)
            sys.exit(1)
        
    args = parser.parse_args()

    # Import heavy backend logic only after arguments are parsed
    try:
        from config import model_opts, xai_opts, fixed_size_models, min_input_size, default_input_size
        from backend.session_manager import SessionManager
        sm = SessionManager(base_dir=os.path.join(GUI_DIR, "sessions"))
        from backend.benchmark_runner import run_benchmark_task
        from backend.exporter import generate_pdf_report, generate_csv_report
        from utils.helpers import build_task_queue
    except ModuleNotFoundError as e:
        print(f"\n[!] Environment Error: {e}")
        print("[!] Please ensure you have activated your virtual environment (e.g., `venv\\Scripts\\activate`)")
        print("[!] and that all dependencies in requirements.txt are installed.\n")
        sys.exit(1)

    # Handle info flags
    if args.list_models:
        print("Available Models:")
        for m in model_opts:
            print(f"  - {m}")
        sys.exit(0)
        
    if args.list_methods:
        print("Available XAI Methods:")
        for x in xai_opts:
            print(f"  - {x}")
        sys.exit(0)
        
    if args.list_metrics:
        print("Available Quality Metrics:")
        metrics = [
            "Gini Index (Sparsity)",
            "Deletion AUC",
            "Insertion AUC",
            "Sensitivity (Max)",
            "Infidelity (Perturbation Faithfulness)"
        ]
        for m in metrics:
            print(f"  - {m}")
        sys.exit(0)

    models = []
    tasks_queue = []
    
    # If a config JSON is provided, load parameters from it
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = json.load(f)
        models = cfg.get("models", [])
        tasks_queue = cfg.get("tasks", [])
        methods = [t.get("method", "Unknown") for t in tasks_queue]
        args.images = cfg.get("images", args.images)
        args.input_sizes = cfg.get("input_sizes", args.input_sizes)
        args.device = cfg.get("device", args.device)
        args.warmups = cfg.get("warmups", args.warmups)
        args.repeats = cfg.get("repeats", args.repeats)
        args.memory_runs = cfg.get("memory_runs", args.memory_runs)
        args.run_order = cfg.get("run_order", args.run_order)
        args.random_seed = cfg.get("random_seed", args.random_seed)
        args.cpu_tdp = cfg.get("cpu_tdp", args.cpu_tdp)
        args.gpu_tdp = cfg.get("gpu_tdp", args.gpu_tdp)
        args.quality_metrics = cfg.get("quality_metrics", args.quality_metrics)
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        for m in methods:
            params = {}
            from config import method_configs
            if m in method_configs:
                for param_key, param_info in method_configs[m].items():
                    if param_info["type"] == "int":
                        params[param_key] = int(param_info["default"])
                    elif param_info["type"] == "float":
                        params[param_key] = float(param_info["default"])
                    else:
                        params[param_key] = param_info["default"]
            tasks_queue.append({"method": m, "params": params})

    sizes = [int(s.strip()) for s in args.input_sizes.split(",") if s.strip()]
    fixed_models = [m for m in models if m in fixed_size_models]
    if fixed_models and (len(sizes) != 1 or sizes[0] != 224):
        print(f"[!] Warning: Fixed-size model(s) selected ({', '.join(fixed_models)}). Input size locked to 224px.")
        sizes = [224]
    images = []
    if args.images.strip():
        for item in args.images.split(","):
            item = item.strip().strip(' "\'')
            if not item: continue
            
            if item.startswith("http://") or item.startswith("https://"):
                images.append(item)
            else:
                resolved_item = resolve_path(item)
                if resolved_item.endswith(".txt") and os.path.isfile(resolved_item):
                    with open(resolved_item, 'r', encoding='utf-8') as f:
                        images.extend([line.strip() for line in f if line.strip()])
                elif os.path.isdir(resolved_item):
                    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                    for root, _, files in os.walk(resolved_item):
                        for file in files:
                            if os.path.splitext(file)[1].lower() in valid_exts:
                                images.append(os.path.join(root, file))
                elif os.path.isfile(resolved_item):
                    images.append(resolved_item)
                else:
                    images.append(item)
    
    if not images:
        print("\n[!] ERROR: No images provided!")
        print("You must provide an image to run the benchmark. You can do this by:")
        print("  1. Passing a local file: --images images/my_image.jpg")
        print("  2. Passing a directory:  --images images/")
        print("  3. Passing a URL:        --images https://example.com/image.jpg")
        print("  4. Passing a text file:  --images image_urls.txt (containing one path/URL per line)\n")
        print("  * You can also mix them using commas: --images images/,url1,image_urls.txt\n")
        sys.exit(1)
    
    quality_metrics = [q.strip() for q in args.quality_metrics.split(",") if q.strip()]
    enable_quality = len(quality_metrics) > 0

    print("\n" + "="*68)
    print("                XAI EFFICIENCY BENCHMARK CLI")
    print("="*68)
    
    from backend.benchmark_runner import collect_environment_metadata
    env_meta = collect_environment_metadata(
        device=args.device,
        custom_cpu_tdp=args.cpu_tdp,
        custom_gpu_tdp=args.gpu_tdp
    )
    
    print(f"Date:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform:     {env_meta.get('platform', platform.platform())}")
    print(f"Python:       {env_meta.get('python_version', 'Unknown')}  |  PyTorch: {env_meta.get('torch_version', torch.__version__)}")
    print(f"CPU:          {env_meta.get('processor', 'Unknown')} (TDP: {args.cpu_tdp if args.cpu_tdp else env_meta.get('cpu_tdp_w', 'N/A')}W)")
    
    gpu_list = env_meta.get('cuda_devices', [])
    if gpu_list:
        for gpu in gpu_list:
            vram_gb = f"{gpu.get('total_memory_mb', 0) / 1024:.2f} GB"
            cc = gpu.get('compute_capability', 'N/A')
            g_tdp = args.gpu_tdp if args.gpu_tdp else gpu.get('tdp_w', 'N/A')
            print(f"GPU [{gpu.get('index', 0)}]:      {gpu.get('name', 'Unknown')} ({vram_gb}, CC {cc}, TDP: {g_tdp}W)")
    elif env_meta.get('mps_available'):
        print(f"GPU:          Apple Silicon GPU (MPS)")
        
    print("-" * 68)
    print(f"Models ({len(models)}):    {', '.join(models)}")
    print(f"Methods ({len(methods)}):   {', '.join(methods)}")
    print(f"Sizes ({len(sizes)}):     {', '.join([f'{s}px' for s in sizes])}")
    print(f"Images ({len(images)}):    {len(images)} loaded")
    
    total_tasks = len(images) * len(models) * len(methods) * len(sizes)
    print(f"Total Tasks:  {total_tasks} execution units ({len(images)} img x {len(models)} mod x {len(methods)} meth x {len(sizes)} res)")
    print(f"Profiling:    {args.warmups} Warmups  |  {args.repeats} Repeats  |  {args.memory_runs} Memory Runs")
    print(f"Task Order:   {args.run_order}" + (f" (Seed: {args.random_seed})" if args.run_order == "Randomized" else ""))
    
    dev_str = args.device.upper()
    active_tdp = args.gpu_tdp if (dev_str in ["CUDA", "MPS"] and args.gpu_tdp) else (args.cpu_tdp if args.cpu_tdp else None)
    if not active_tdp:
        if dev_str in ["CUDA", "MPS"] and gpu_list:
            active_tdp = f"{gpu_list[0].get('tdp_w', 'N/A')}W"
        else:
            active_tdp = f"{env_meta.get('cpu_tdp_w', 'N/A')}W"
    else:
        active_tdp = f"{active_tdp}W"
    print(f"Device:       {dev_str} (Active Energy TDP: {active_tdp})")
    
    if enable_quality:
        print(f"Quality Eval: Enabled ({len(quality_metrics)} metrics: {', '.join(quality_metrics)})")
    else:
        print(f"Quality Eval: Disabled (Execution time profiling only)")
        
    print("\n" + "-" * 68)
    print("METHOD PARAMETERS MAPPING")
    print("-" * 68)
    print(f"{'Method Name':<25} | {'Base Algorithm':<20} | {'Configuration'}")
    print("-" * 68)
    for t in tasks_queue:
        p_dict = t.get("params", {})
        p_str = ", ".join([f"{k}: {v}" for k, v in p_dict.items()]) if p_dict else "Standard / Default"
        disp_m = t.get("method", "Unknown")
        base_algo = str(t.get("base_name") or (disp_m.split("_")[0] if "_" in disp_m else disp_m)).replace("_", " ").title()
        print(f"{disp_m:<25} | {base_algo:<20} | {p_str}")
        
    print("="*68 + "\n")

    start_time = time.time()

    # Initialize a new batch using SessionManager
    batch_id = sm.start_batch()
    print(f"[*] Creating batch: {batch_id}")
    
    batch_config = {
        "batch_id": batch_id,
        "warmup_runs": args.warmups,
        "memory_runs": args.memory_runs,
        "repeat_count": args.repeats,
        "enable_quality_metrics": enable_quality,
        "selected_quality_metrics": quality_metrics,
        "run_order": args.run_order,
        "random_seed": args.random_seed,
        "models": models,
        "input_sizes": [str(s) for s in sizes],
        "methods": methods,
        "methods_info": tasks_queue,
        "device_mode": args.device,
        "image_sources": images,
        "started_at": datetime.now().isoformat(),
        "custom_cpu_tdp": args.cpu_tdp,
        "custom_gpu_tdp": args.gpu_tdp
    }
    sm.save_batch_config(batch_id, batch_config)

    # Prepare Global Results Structure
    global_results = []

    # Flatten Task Queue using build_task_queue
    formatted_methods = [
        {
            "display_name": t["method"],
            "base_name": t["method"].split("_")[0] if "_" in t["method"] else t["method"],
            "params": t.get("params", {})
        }
        for t in tasks_queue
    ]

    seed = args.random_seed if args.run_order == "Randomized" else None
    task_queue = build_task_queue(
        num_images=len(images),
        models=models,
        sizes=sizes,
        methods=formatted_methods,
        run_order=args.run_order,
        seed=seed
    )

    total_tasks = len(task_queue)
    for task_idx, task in enumerate(task_queue, 1):
        img_i = task["img_i"]
        src_img = images[img_i]
        model_name = task["model_name"]
        size = task["target_size"]
        method_name = task["method_name"]
        method_params = task.get("method_params", {})

        is_url = src_img.startswith("http://") or src_img.startswith("https://")
        if not is_url and not os.path.exists(src_img):
            print(f"[!] Warning: Local image {src_img} not found. Skipping.")
            continue

        model_entry = get_or_create_cli_model_entry(
            global_results, img_i, src_img, model_name, size, sm, batch_id, args
        )

        print(f"\n[{task_idx}/{total_tasks}] Executing {method_name} on {model_name} ({size}px)")

        config = {
            "model_name": model_name, 
            "image_source": model_entry["src_path"], 
            "methods": [method_name], 
            "method_params": method_params,
            "force_device": args.device if args.device != "auto" else None, 
            "input_size": size,
            "warmup_runs": args.warmups,
            "memory_runs": args.memory_runs,
            "repeat_count": args.repeats,
            "run_order": args.run_order,
            "enable_quality_metrics": enable_quality,
            "selected_quality_metrics": quality_metrics,
            "custom_cpu_tdp": args.cpu_tdp,
            "custom_gpu_tdp": args.gpu_tdp
        }

        res = run_benchmark_task(config, model_entry["session_dir"])
        if res:
            model_entry["results"].extend(res)
            for r in res:
                if str(r.get("Status", "")).startswith("Failed"):
                    print(f"  -> [!] {r.get('Method', 'Unknown')}: Execution failed ({r.get('Status')})")
                    continue
                out_msg = f"  -> {r.get('Method', 'Unknown')}: {r.get('Runtime (sec)', 'N/A')} sec | {r.get('Peak Memory (MB)', 'N/A')} MB | {r.get('Estimated Energy Consumption (kWh)', 'N/A')} kWh"
                qm_parts = []
                for qm in ["Gini Index", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)", "Infidelity"]:
                    if qm in r and r[qm] not in [None, "N/A"]:
                        qm_parts.append(f"{qm}: {r[qm]}")
                if qm_parts:
                    out_msg += f" | {', '.join(qm_parts)}"
                print(out_msg)

            heatmap_path = os.path.abspath(os.path.join(model_entry["session_dir"], "heatmaps", f"{method_name}.png"))
            if os.path.exists(heatmap_path):
                print(f"     Heatmap: file:///{heatmap_path.replace(chr(92), '/')}")
        else:
            print(f"  -> [!] Task failed or returned no results.")

    total_execution_time = time.time() - start_time

    # Ensure global_results is sorted by image index
    global_results.sort(key=lambda g: int(g["image_id"].split()[1]) if "Image " in g["image_id"] else 0)

    # Save Batch Results JSON (GUI-compatible so run appears in History viewer)
    final_json_path = os.path.join(sm.base_dir, batch_id, "batch_results.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": global_results,
            "methods": methods,
            "benchmark_settings": batch_config,
            "environment": env_meta,
            "started_at": batch_config["started_at"],
            "completed_at": datetime.now().isoformat(),
            "total_execution_time": round(total_execution_time, 2),
            "methods_info": formatted_methods
        }, f, indent=4)
        
    # Generate CSV and PDF reports
    csv_path = os.path.join(sm.base_dir, batch_id, "batch_summary.csv")
    pdf_path = os.path.join(sm.base_dir, batch_id, "report.pdf")
    try:
        generate_csv_report(global_results, csv_path)
    except Exception as e:
        print(f"[!] Warning: Could not compile CSV summary: {e}")
        
    try:
        generate_pdf_report(
            batch_id=batch_id,
            results_data=global_results,
            selected_methods=methods,
            output_path=pdf_path,
            total_time=round(total_execution_time, 2),
            environment=env_meta,
            benchmark_settings=batch_config
        )
    except Exception as e:
        print(f"[!] Warning: Could not compile PDF report: {e}")

    # Format helper for terminal tables
    def format_cli_cell(val, fmt=None):
        if val is None or pd.isna(val) or val == "N/A" or val == "-":
            return "-"
        if fmt:
            try:
                return fmt.format(float(val))
            except (ValueError, TypeError):
                return str(val)
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    def print_cli_table(title, headers, rows):
        print("\n" + "="*68)
        print(title)
        print("="*68)
        col_strs = [f"{h:<{w}}" for h, w, _ in headers]
        hdr_line = " | ".join(col_strs)
        print(hdr_line)
        print("-" * len(hdr_line))
        for r in rows:
            r_strs = []
            for (h, w, fmt), v in zip(headers, r):
                f_val = format_cli_cell(v, fmt)
                r_strs.append(f"{f_val:<{w}}")
            print(" | ".join(r_strs))

    # Ensure Energy Consumption (kWh) is calculated across all results
    active_tdp_val = args.gpu_tdp if (args.device.upper() in ["CUDA", "MPS"] and args.gpu_tdp) else (args.cpu_tdp if args.cpu_tdp else None)
    if active_tdp_val is None:
        if args.device.upper() in ["CUDA", "MPS"] and env_meta.get("cuda_devices"):
            active_tdp_val = env_meta["cuda_devices"][0].get("tdp_w")
        else:
            active_tdp_val = env_meta.get("cpu_tdp_w")

    # Compile flat DataFrame for multi-dimensional analytics
    all_rows = []
    for g_idx, img_g in enumerate(global_results):
        for mod in img_g.get("models", []):
            for res in mod.get("results", []):
                if active_tdp_val is not None:
                    rt = res.get("Runtime (sec)") or res.get("Attribution Runtime (sec)")
                    en = res.get("Estimated Energy Consumption (kWh)") or res.get("Estimated Energy Consumption (kW)")
                    if rt is not None and (en is None or pd.isna(en)):
                        calc_en = round((float(rt) * float(active_tdp_val)) / (3600.0 * 1000.0), 8)
                        res["Estimated Energy Consumption (kWh)"] = calc_en
                        res["Estimated Energy Consumption (kW)"] = calc_en
                r_copy = res.copy()
                r_copy["Image Index"] = g_idx
                if "Resolution" not in r_copy or not r_copy["Resolution"]:
                    r_copy["Resolution"] = f"{mod.get('input_size', 224)}x{mod.get('input_size', 224)}"
                all_rows.append(r_copy)

    if all_rows:
        fdf = normalize_metric_columns(pd.DataFrame(all_rows))
        runtime_col = metric_col(fdf, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
        memory_col = metric_col(fdf, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)

        n_methods = fdf["Method"].nunique()
        n_models = fdf["Model"].nunique()
        n_resolutions = fdf["Resolution"].nunique() if "Resolution" in fdf.columns else 1

        qm_defs = [
            ("Gini Index", "Gini", 7, "{:.3f}"),
            ("Deletion AUC", "Del AUC", 9, "{:.3f}"),
            ("Insertion AUC", "Ins AUC", 9, "{:.3f}"),
            ("Sensitivity (Max)", "Sens (Max)", 11, "{:.3f}"),
            ("Infidelity", "Infidelity", 11, "{:.3e}"),
        ]
        active_qm = [q for q in qm_defs if q[0] in fdf.columns and fdf[q[0]].notna().any()]

        # 1. Configuration Averages
        group_cols = ["Method", "Model"]
        if n_resolutions > 1:
            group_cols.append("Resolution")

        n_images = int(fdf["Image Index"].nunique()) if "Image Index" in fdf.columns else 1
        std_runtime_src = "Attribution Runtime Std (sec)" if (n_images == 1 and "Attribution Runtime Std (sec)" in fdf.columns) else runtime_col
        std_runtime_func = "mean" if n_images == 1 else "std"

        std_mem_src = "Attribution Memory Std (MB)" if (n_images == 1 and "Attribution Memory Std (MB)" in fdf.columns) else memory_col
        std_mem_func = "mean" if n_images == 1 else "std"

        agg_dict_config = {
            "Mean Runtime (s)": (runtime_col, "mean"),
            "Runtime Std (s)": (std_runtime_src, std_runtime_func),
        }
        if "Estimated Energy Consumption (kWh)" in fdf.columns and fdf["Estimated Energy Consumption (kWh)"].notna().any():
            agg_dict_config["Energy (kWh)"] = ("Estimated Energy Consumption (kWh)", "mean")
        agg_dict_config["Peak Mem (MB)"] = (memory_col, "mean")
        agg_dict_config["Peak Mem Std (MB)"] = (std_mem_src, std_mem_func)
        for col_name, short_name, _, _ in active_qm:
            agg_dict_config[short_name] = (col_name, "mean")
        agg_dict_config["Samples"] = (runtime_col, "count")

        config_avg_df = fdf.groupby(group_cols, sort=False).agg(**agg_dict_config).reset_index()

        headers_cfg = [("Method", 20, None), ("Model", 14, None)]
        if n_resolutions > 1:
            headers_cfg.append(("Resolution", 10, None))
        headers_cfg.extend([
            ("Runtime (s)", 12, "{:.4f}"),
            ("Runtime Std (s)", 15, "{:.4f}"),
            ("Peak Mem (MB)", 13, "{:.2f}"),
            ("Peak Mem Std (MB)", 17, "{:.2f}"),
            ("Energy (kWh)", 12, "{:.2e}"),
        ])
        for _, short_name, w, fmt in active_qm:
            headers_cfg.append((short_name, w, fmt))
        headers_cfg.append(("Samples", 7, "{:d}"))

        rows_cfg = []
        for _, r in config_avg_df.iterrows():
            row_vals = [r.get("Method"), r.get("Model")]
            if n_resolutions > 1:
                row_vals.append(r.get("Resolution"))
            row_vals.extend([
                r.get("Mean Runtime (s)"),
                r.get("Runtime Std (s)"),
                r.get("Peak Mem (MB)"),
                r.get("Peak Mem Std (MB)"),
                r.get("Energy (kWh)"),
            ])
            for _, short_name, _, _ in active_qm:
                row_vals.append(r.get(short_name))
            row_vals.append(int(r.get("Samples", 1)))
            rows_cfg.append(row_vals)

        print_cli_table("CONFIGURATION AVERAGES", headers_cfg, rows_cfg)

        # 2. XAI Method Comparison (when multiple methods benchmarked)
        if n_methods > 1:
            meth_summary = method_detail_summary(fdf)
            headers_meth = [
                ("Method", 20, None),
                ("Mean Runtime (s)", 16, "{:.4f}"),
                ("Runtime Std (s)", 15, "{:.4f}"),
                ("Peak Mem (MB)", 13, "{:.2f}"),
                ("Energy (kWh)", 12, "{:.2e}"),
            ]
            for _, short_name, w, fmt in active_qm:
                headers_meth.append((short_name, w, fmt))
            headers_meth.append(("Samples", 7, "{:d}"))
            
            rows_meth = []
            for _, r in meth_summary.iterrows():
                r_vals = [
                    r.get("Method"),
                    r.get("Mean Attribution Runtime (sec)"),
                    r.get("Attribution Runtime Std (sec)"),
                    r.get("Mean Peak Attribution Memory (MB)"),
                    r.get("Mean Estimated Energy Consumption (kWh)"),
                ]
                for col_name, _, _, _ in active_qm:
                    r_vals.append(r.get(f"Mean {col_name}"))
                r_vals.append(int(r.get("Samples", 1)))
                rows_meth.append(r_vals)
            print_cli_table("XAI METHOD COMPARISON", headers_meth, rows_meth)

        # 3. Pareto Analysis Summary (when quality metrics evaluated & multiple methods)
        if n_methods > 1 and len(active_qm) > 0:
            ranking_df = compute_pareto_ranking(fdf, runtime_col)
            if not ranking_df.empty:
                pareto_cols = [c for c in ranking_df.columns if c != "Method" and c != "Overall"]
                headers_pareto = [("Method", 20, None)]
                for c in pareto_cols:
                    clean_c = c.replace("Runtime-", "vs ").replace("Runtime–", "vs ")
                    headers_pareto.append((clean_c, 18, None))
                headers_pareto.append(("Overall (Frontiers)", 19, None))
                
                rows_pareto = []
                for _, r in ranking_df.iterrows():
                    r_vals = [r.get("Method")]
                    for c in pareto_cols:
                        r_vals.append(r.get(c))
                    r_vals.append(r.get("Overall"))
                    rows_pareto.append(r_vals)
                print_cli_table("PARETO ANALYSIS SUMMARY", headers_pareto, rows_pareto)

        # 4. Model Comparison (when multiple model architectures benchmarked)
        if n_models > 1:
            model_agg = {
                "Mean Runtime (s)": (runtime_col, "mean"),
                "Peak Mem (MB)": (memory_col, "mean"),
            }
            if "Estimated Energy Consumption (kWh)" in fdf.columns and fdf["Estimated Energy Consumption (kWh)"].notna().any():
                model_agg["Energy (kWh)"] = ("Estimated Energy Consumption (kWh)", "mean")
            for col_name, short_name, _, _ in active_qm:
                model_agg[short_name] = (col_name, "mean")
            model_agg["Samples"] = (runtime_col, "count")
            model_summary = fdf.groupby("Model", sort=False).agg(**model_agg).reset_index().sort_values("Mean Runtime (s)")
            
            headers_mod = [
                ("Model", 16, None),
                ("Mean Runtime (s)", 16, "{:.4f}"),
                ("Peak Mem (MB)", 13, "{:.2f}"),
                ("Energy (kWh)", 12, "{:.2e}"),
            ]
            for _, short_name, w, fmt in active_qm:
                headers_mod.append((short_name, w, fmt))
            headers_mod.append(("Samples", 7, "{:d}"))
            
            rows_mod = []
            for _, r in model_summary.iterrows():
                r_vals = [
                    r.get("Model"),
                    r.get("Mean Runtime (s)"),
                    r.get("Peak Mem (MB)"),
                    r.get("Energy (kWh)"),
                ]
                for _, short_name, _, _ in active_qm:
                    r_vals.append(r.get(short_name))
                r_vals.append(int(r.get("Samples", 1)))
                rows_mod.append(r_vals)
            print_cli_table("MODEL COMPARISON", headers_mod, rows_mod)

        # 5. Resolution Comparison (only when multiple image sizes benchmarked)
        if n_resolutions > 1:
            size_summary = image_size_summary(fdf)
            if not size_summary.empty:
                headers_res = [
                    ("Resolution", 12, None),
                    ("Mean Runtime (s)", 16, "{:.4f}"),
                    ("Runtime Std (s)", 15, "{:.4f}"),
                    ("Peak Mem (MB)", 13, "{:.2f}"),
                    ("Energy (kWh)", 12, "{:.2e}"),
                    ("Samples", 7, "{:d}"),
                ]
                rows_res = []
                for _, r in size_summary.iterrows():
                    res_raw = str(r.get("Resolution", ""))
                    res_lbl = f"{int(round(float(res_raw)))}px" if (res_raw.replace('.', '', 1).isdigit()) else res_raw
                    rows_res.append([
                        res_lbl,
                        r.get("Mean Attribution Runtime (sec)"),
                        r.get("Attribution Runtime Std (sec)"),
                        r.get("Mean Peak Attribution Memory (MB)"),
                        r.get("Mean Estimated Energy Consumption (kWh)"),
                        int(r.get("Samples", 1)),
                    ])
                print_cli_table("RESOLUTION COMPARISON", headers_res, rows_res)
        
    print("\n" + "="*68)
    print(f"[+] Benchmark Complete! (Total Duration: {total_execution_time:.2f}s)")
    print(f"   Batch JSON:    file:///{os.path.abspath(final_json_path).replace(chr(92), '/')}")
    if os.path.exists(csv_path):
        print(f"   CSV Export:    file:///{os.path.abspath(csv_path).replace(chr(92), '/')}")
    if os.path.exists(pdf_path):
        print(f"   PDF Report:    file:///{os.path.abspath(pdf_path).replace(chr(92), '/')}")
    print(f"   Tip: Open Streamlit UI (`App.bat`) to inspect full interactive charts in History!")
    print("="*68 + "\n")

if __name__ == "__main__":
    main()
