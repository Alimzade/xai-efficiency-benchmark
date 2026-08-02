#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
from datetime import datetime

# Setup paths so imports from 'gui.config' and 'models' work regardless of where cli.py is executed from
CLI_DIR = os.path.dirname(os.path.abspath(__file__))
GUI_DIR = os.path.abspath(os.path.join(CLI_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(GUI_DIR, ".."))
sys.path.append(PROJECT_ROOT)

# Heavy imports are deferred to main() to allow --help to run instantly



def run_interactive_wizard():
    try:
        import questionary
    except ImportError:
        print("\n[!] The interactive wizard requires the 'questionary' library.")
        print("[!] Please run: pip install questionary")
        return False

    try:
        from config import model_opts, xai_opts
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
                import json
                with open(last_run_file, "r") as f:
                    cfg = json.load(f)
            except Exception as e:
                print(f"[!] Could not load previous config: {e}")
                sys.exit(1)
                
            input_type = questionary.select(
                "How would you like to provide new images?",
                choices=[
                    "Scan a directory for images",
                    "Provide a single local file path",
                    "Provide a single URL",
                    "Provide a .txt file containing multiple paths/URLs"
                ]
            ).ask()
            if not input_type: sys.exit(0)

            images_input = ""
            if "directory" in input_type:
                default_dir = os.path.join(CLI_DIR, "images") if os.path.exists(os.path.join(CLI_DIR, "images")) else "gui/images/"
                images_input = questionary.text(
                    "Enter the directory path (e.g. gui/images/):",
                    default=default_dir,
                    validate=lambda x: os.path.isdir(x) or "Please enter a valid directory path."
                ).ask()
            elif "local file" in input_type:
                images_input = questionary.text(
                    "Enter the local file path (e.g. gui/images/my_image.jpg):",
                    validate=lambda x: os.path.isfile(x) or "Please enter a valid file path."
                ).ask()
            elif "URL" in input_type:
                images_input = questionary.text(
                    "Enter the URL:",
                    validate=lambda x: x.startswith("http") or "URL must start with http:// or https://"
                ).ask()
            elif ".txt" in input_type:
                images_input = questionary.text(
                    "Enter the path to your .txt file (e.g. gui/image_urls.txt):",
                    validate=lambda x: (x.endswith(".txt") and os.path.isfile(x)) or "Please enter a valid .txt file path."
                ).ask()
                
            if not images_input: sys.exit(0)
            
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
    
    input_type = questionary.select(
        "How would you like to provide images?",
        choices=[
            "Scan a directory for images",
            "Provide a single local file path",
            "Provide a single URL",
            "Provide a .txt file containing multiple paths/URLs"
        ]
    ).ask()
    if not input_type: sys.exit(0)

    images_input = ""
    if "directory" in input_type:
        default_dir = os.path.join(CLI_DIR, "images") if os.path.exists(os.path.join(CLI_DIR, "images")) else "gui/images/"
        images_input = questionary.text(
            "Enter the directory path (e.g. gui/images/):",
            default=default_dir,
            validate=lambda x: os.path.isdir(x) or "Please enter a valid directory path."
        ).ask()
    elif "local file" in input_type:
        images_input = questionary.text(
            "Enter the local file path (e.g. gui/images/my_image.jpg):",
            validate=lambda x: os.path.isfile(x) or "Please enter a valid file path."
        ).ask()
    elif "URL" in input_type:
        images_input = questionary.text(
            "Enter the URL:",
            validate=lambda x: x.startswith("http") or "URL must start with http:// or https://"
        ).ask()
    elif ".txt" in input_type:
        images_input = questionary.text(
            "Enter the path to your .txt file (e.g. gui/image_urls.txt):",
            validate=lambda x: (x.endswith(".txt") and os.path.isfile(x)) or "Please enter a valid .txt file path."
        ).ask()
        
    if not images_input: sys.exit(0)

    sizes_input = questionary.text(
        "Input sizes in pixels (comma-separated):",
        default="224"
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
    
    device_opt = questionary.select(
        "Which device to force execution on?",
        choices=["auto", "cpu", "cuda", "mps"]
    ).ask()
    if device_opt is None: sys.exit(0)
    
    cpu_tdp_opt = questionary.text(
        "Override CPU TDP (W)? [Leave blank to use detected]:"
    ).ask()
    if cpu_tdp_opt is None: sys.exit(0)
    
    if gpu_list:
        gpu_tdp_opt = questionary.text(
            "Override GPU TDP (W)? [Leave blank to use detected]:"
        ).ask()
        if gpu_tdp_opt is None: sys.exit(0)

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

    print("\n[*] Setup Complete! Initializing backend...\n")
    
    # Save JSON config to pass to CLI and for future restore
    config_file = os.path.join(CLI_DIR, "assets", ".last_run_config.json")
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    import json
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
            "gpu_tdp": float(gpu_tdp_opt) if gpu_tdp_opt.strip() else None
        }, f)
    
    # Inject user selections into sys.argv so argparse parses them seamlessly
    sys.argv.extend([
        "--config", config_file
    ])
    return True

def get_or_create_cli_model_entry(global_results, img_i, src_img, model_name, size, sm, batch_id, args):
    img_id = f"Image {img_i+1}"
    img_group = next((g for g in global_results if g["image_id"] == img_id), None)
    if not img_group:
        img_group = {
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
        from config import model_opts, xai_opts
        from backend.session_manager import SessionManager
        sm = SessionManager(base_dir=os.path.join(CLI_DIR, "sessions"))
        from backend.benchmark_runner import run_benchmark_task
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
        metrics = ["Gini Index (Sparsity)", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)"]
        for m in metrics:
            print(f"  - {m}")
        sys.exit(0)

    models = []
    tasks_queue = []
    
    # If a config JSON is provided, load parameters from it
    if args.config and os.path.exists(args.config):
        import json
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
    images = []
    if args.images.strip():
        for item in args.images.split(","):
            item = item.strip().strip(' "\'')
            if not item: continue
            
            if item.startswith("http://") or item.startswith("https://"):
                images.append(item)
            elif item.endswith(".txt") and os.path.exists(item):
                with open(item, 'r', encoding='utf-8') as f:
                    images.extend([line.strip() for line in f if line.strip()])
            elif os.path.isdir(item):
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                for root, _, files in os.walk(item):
                    for file in files:
                        if os.path.splitext(file)[1].lower() in valid_exts:
                            images.append(os.path.join(root, file))
            else:
                images.append(item)
    
    if not images:
        print("\n[!] ERROR: No images provided!")
        print("You must provide an image to run the benchmark. You can do this by:")
        print("  1. Passing a local file: --images gui\\images\\my_image.jpg")
        print("  2. Passing a directory:  --images gui\\images\\")
        print("  3. Passing a URL:        --images https://example.com/image.jpg")
        print("  4. Passing a text file:  --images gui\\image_urls.txt (containing one path/URL per line)\n")
        print("  * You can also mix them using commas: --images gui\\images\\,url1,gui\\image_urls.txt\n")
        sys.exit(1)
    
    quality_metrics = [q.strip() for q in args.quality_metrics.split(",") if q.strip()]
    enable_quality = len(quality_metrics) > 0

    print("\n" + "="*60)
    print("🚀 XAI EFFICIENCY BENCHMARK CLI 🚀")
    print("="*60)
    
    from backend.benchmark_runner import collect_environment_metadata
    from datetime import datetime
    env_meta = collect_environment_metadata()
    
    print(f"Date:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python:       {env_meta.get('python_version', 'Unknown')}")
    print(f"CPU:          {env_meta.get('processor', 'Unknown')} (TDP: {args.cpu_tdp if args.cpu_tdp else env_meta.get('cpu_tdp_w', 'N/A')}W)")
    
    gpu_list = env_meta.get('cuda_devices', [])
    if gpu_list:
        gpu = gpu_list[0]
        print(f"GPU:          {gpu.get('name', 'Unknown')} (TDP: {args.gpu_tdp if args.gpu_tdp else gpu.get('tdp_w', 'N/A')}W)")
        
    print("-" * 60)
    print(f"Models:       {models}")
    print(f"Methods:      {methods}")
    print(f"Sizes:        {sizes}")
    print(f"Images:       {len(images)} loaded")
    print(f"Execution:    {args.warmups} Warmups | {args.repeats} Repeats | {args.memory_runs} Mem Runs")
    print(f"Task Order:   {args.run_order}" + (f" (Seed: {args.random_seed})" if args.run_order == "Randomized" else ""))
    print(f"Device:       {args.device.upper()}")
    print("="*60 + "\n")

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
                print(f"  -> {r.get('Method', 'Unknown')}: {r.get('Runtime (sec)', 'N/A')} sec | {r.get('Peak Memory (MB)', 'N/A')} MB | {r.get('Estimated Energy Consumption (kW)', 'N/A')} kW")

            heatmap_path = os.path.abspath(os.path.join(model_entry["session_dir"], "heatmaps", f"{method_name}.png"))
            if os.path.exists(heatmap_path):
                print(f"     Heatmap: file:///{heatmap_path.replace(chr(92), '/')}")
        else:
            print(f"  -> [!] Task failed or returned no results.")

    # Ensure global_results is sorted by image index
    global_results.sort(key=lambda g: int(g["image_id"].split()[1]) if "Image " in g["image_id"] else 0)

    # Save Global Results JSON
    final_json_path = os.path.join(sm.base_dir, batch_id, "results.json")
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(global_results, f, indent=4)
        
    print("\n" + "="*60)
    print("📊 BENCHMARK SUMMARY (AVERAGES)")
    print("="*60)
    method_stats = {}
    for img_g in global_results:
        for mod in img_g.get("models", []):
            for res in mod.get("results", []):
                m_name = res.get("Method", "Unknown")
                if m_name not in method_stats:
                    method_stats[m_name] = {"runtime": [], "memory": [], "energy": []}
                
                # Check for numerical values, ignoring 'N/A'
                try:
                    if res.get("Runtime (sec)") not in [None, "N/A"]:
                        method_stats[m_name]["runtime"].append(float(res["Runtime (sec)"]))
                    if res.get("Peak Memory (MB)") not in [None, "N/A"]:
                        method_stats[m_name]["memory"].append(float(res["Peak Memory (MB)"]))
                    if res.get("Estimated Energy Consumption (kW)") not in [None, "N/A"]:
                        method_stats[m_name]["energy"].append(float(res["Estimated Energy Consumption (kW)"]))
                except ValueError: pass

    print(f"{'Method':<25} | {'Runtime (s)':<12} | {'Memory (MB)':<12} | {'Energy (kW)':<12}")
    print("-" * 68)
    for m_name, stats in method_stats.items():
        avg_r = f"{sum(stats['runtime'])/len(stats['runtime']):.4f}" if stats["runtime"] else "N/A"
        avg_m = f"{sum(stats['memory'])/len(stats['memory']):.2f}" if stats["memory"] else "N/A"
        avg_e = f"{sum(stats['energy'])/len(stats['energy']):.2e}" if stats["energy"] else "N/A"
        print(f"{m_name:<25} | {avg_r:<12} | {avg_m:<12} | {avg_e:<12}")
        
    print("\n" + "="*60)
    print(f"🎉 Benchmark Complete! Results compiled at:")
    abs_json = os.path.abspath(final_json_path)
    print(f"   file:///{abs_json.replace(chr(92), '/')}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
