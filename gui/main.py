"""
Main entry point for the Streamlit-based GUI application.
Handles page routing, UI initialization, and overall application state.
"""
import os
import re
import time
import json
import logging
import warnings

# --- SILENCE KNOWN WARNINGS (targeted only) ---
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="captum.*")
warnings.filterwarnings("ignore", message=".*components.v1.html.*")
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*Passing.*palette.*without assigning.*hue.*")

import streamlit as st
# Silence sub-logger warnings that get reset during streamlit import
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

import streamlit.components.v1 as components

# --- APP CONFIGURATION ---
st.set_page_config(page_title="XAI Efficiency Benchmark", page_icon="ðŸ”", layout="wide")

# --- INITIALIZATION SPLASH SCREEN ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    def render_splash(step):
        def get_status(item_step, active_step):
            if item_step < active_step:
                return '<div class="task-icon check-glow">✓</div>', 'color: #34d399; font-weight: 500; opacity: 1;'
            elif item_step == active_step:
                return '<div class="task-icon"><div class="pulse-loader"></div></div>', 'color: #38bdf8; font-weight: 600; opacity: 1;'
            else:
                return '<div class="task-icon pending"></div>', 'color: #475569; opacity: 0.4;'
        status1, style1 = get_status(1, step)
        status2, style2 = get_status(2, step)
        status3, style3 = get_status(3, step)
        status4, style4 = get_status(4, step)
        
        opacity = 1 if step == 5 else 0
        launching_html = f"""<div style="margin-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 1rem; opacity: {opacity}; transition: opacity 0.3s ease;">
    <div class="flicker-text">Starting Benchmark...</div>
</div>"""

        splash_path = os.path.join(os.path.dirname(__file__), "assets", "splash.html")
        with open(splash_path, "r", encoding="utf-8") as f:
            splash_template = f.read()
        return splash_template.replace('data-style="style1"', f'style="{style1}"').replace('<!-- status1 -->', status1) \
                              .replace('data-style="style2"', f'style="{style2}"').replace('<!-- status2 -->', status2) \
                              .replace('data-style="style3"', f'style="{style3}"').replace('<!-- status3 -->', status3) \
                              .replace('data-style="style4"', f'style="{style4}"').replace('<!-- status4 -->', status4) \
                              .replace('<!-- launching_html -->', launching_html)

    placeholder = st.empty()

    # --- STEP 1: LOAD FRAMEWORKS ---
    placeholder.markdown(render_splash(1), unsafe_allow_html=True)
    import pyarrow
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    time.sleep(0.4)

    # --- STEP 2: LOAD XAI CORE ---
    placeholder.markdown(render_splash(2), unsafe_allow_html=True)
    from config import sm, default_device_mode
    time.sleep(0.4)

    # --- STEP 3: SYSTEM CHECK ---
    placeholder.markdown(render_splash(3), unsafe_allow_html=True)
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    time.sleep(0.1)

    # --- STEP 4: WORKSPACE CHECK ---
    placeholder.markdown(render_splash(4), unsafe_allow_html=True)
    from views.configure import render_configure_page
    from views.active_run import render_active_run_page
    from views.history import render_history_page
    from views.documentation import render_documentation_page
    time.sleep(0.1)

    # --- STEP 5: FINAL LAUNCH TRANSITION ---
    placeholder.markdown(render_splash(5), unsafe_allow_html=True)
    time.sleep(0.4)

    st.session_state.initialized = True
    st.rerun()

# --- MAIN SCRIPTS TOP-LEVEL IMPORTS (Instantaneous on rerun!) ---
import pyarrow
import pandas as pd
import torch
import matplotlib.pyplot as plt
from config import sm, default_device_mode, model_opts, xai_opts, PROJECT_ROOT
from backend.benchmark_runner import run_benchmark_task
from utils.helpers import get_device_string, timestamp_now
from utils.state import get_or_create_result_group, write_current_batch_results_json
from views.configure import render_configure_page
from views.active_run import render_active_run_page
from views.history import render_history_page
from views.documentation import render_documentation_page

# CSS
css_path = os.path.join(PROJECT_ROOT, "gui", "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
    <div class="app-title">
        <h1>XAI Efficiency Benchmark</h1>
        <p>Compare attribution runtime, peak memory, and image-size behavior across models, methods and hardwares.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Initialize Session State ---
if 'selected_history_batch' not in st.session_state: st.session_state.selected_history_batch = None
if 'first_run_render' not in st.session_state: st.session_state.first_run_render = False
if 'img_idx' not in st.session_state: st.session_state.img_idx = 0
if 'persisted_urls' not in st.session_state: st.session_state.persisted_urls = ""
if 'last_run_results' not in st.session_state: st.session_state.last_run_results = []
if 'stop_requested' not in st.session_state: st.session_state.stop_requested = False
if 'is_finished' not in st.session_state: st.session_state.is_finished = False
if 'benchmark_running' not in st.session_state: st.session_state.benchmark_running = False
if 'run_progress_idx' not in st.session_state: st.session_state.run_progress_idx = 0
if 'current_batch_id' not in st.session_state: st.session_state.current_batch_id = ""
if 'last_run_batch_id' not in st.session_state: st.session_state.last_run_batch_id = ""
if 'completed_batch_id' not in st.session_state: st.session_state.completed_batch_id = ""
if 'completion_notice_batch_id' not in st.session_state: st.session_state.completion_notice_batch_id = ""
if 'benchmark_ready_to_run' not in st.session_state: st.session_state.benchmark_ready_to_run = False
if 'current_img_base64' not in st.session_state: st.session_state.current_img_base64 = ""
if 'batch_start_time' not in st.session_state: st.session_state.batch_start_time = None
if 'batch_started_at' not in st.session_state: st.session_state.batch_started_at = ""
if 'batch_completed_at' not in st.session_state: st.session_state.batch_completed_at = ""
if 'total_execution_time' not in st.session_state: st.session_state.total_execution_time = 0
if 'task_queue' not in st.session_state: st.session_state.task_queue = []
if 'prepared_img_sources' not in st.session_state: st.session_state.prepared_img_sources = []
if 'current_run_order' not in st.session_state: st.session_state.current_run_order = "Balanced"
if 'current_batch_methods' not in st.session_state: st.session_state.current_batch_methods = []
if 'current_batch_models' not in st.session_state: st.session_state.current_batch_models = []
if 'current_batch_sizes' not in st.session_state: st.session_state.current_batch_sizes = []
if 'selected_models' not in st.session_state: st.session_state.selected_models = ["resnet50"]
if 'input_sizes_options' not in st.session_state:
    st.session_state.input_sizes_options = [
        "32", "64", "96", "128", "160", "192", "224", "256", "288", "299", 
        "320", "352", "384", "416", "448", "480", "512", "576", "640", "704", 
        "768", "832", "896", "960", "1024"
    ]
if 'selected_input_sizes' not in st.session_state:
    st.session_state.selected_input_sizes = ["224"]
if 'input_size_str' not in st.session_state:
    st.session_state.input_size_str = "224"
if 'selected_methods' not in st.session_state: st.session_state.selected_methods = ["Saliency", "Integrated_Gradients"]
if 'selected_warmups' not in st.session_state: st.session_state.selected_warmups = 3
if 'selected_repeats' not in st.session_state: st.session_state.selected_repeats = 5
if 'selected_memory_runs' not in st.session_state: st.session_state.selected_memory_runs = 1
if 'selected_run_order' not in st.session_state: st.session_state.selected_run_order = "Balanced"
if 'selected_random_seed' not in st.session_state: st.session_state.selected_random_seed = 42
if 'current_random_seed' not in st.session_state: st.session_state.current_random_seed = 42
if 'enable_quality_metrics' not in st.session_state: st.session_state.enable_quality_metrics = False
if 'selected_quality_metrics' not in st.session_state: st.session_state.selected_quality_metrics = ["Gini Index (Sparsity)", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)"]

if 'selected_device_mode' not in st.session_state: st.session_state.selected_device_mode = default_device_mode
if 'current_device_mode' not in st.session_state: st.session_state.current_device_mode = default_device_mode
if 'current_warmups' not in st.session_state: st.session_state.current_warmups = 1
if 'current_repeats' not in st.session_state: st.session_state.current_repeats = 5
if 'current_memory_runs' not in st.session_state: st.session_state.current_memory_runs = 1
if 'current_enable_quality_metrics' not in st.session_state: st.session_state.current_enable_quality_metrics = False
if 'current_selected_quality_metrics' not in st.session_state: st.session_state.current_selected_quality_metrics = ["Gini Index (Sparsity)", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)"]
if 'current_page' not in st.session_state: st.session_state.current_page = "Configure"

if 'custom_cpu_tdp' not in st.session_state: st.session_state.custom_cpu_tdp = None
if 'custom_gpu_tdp' not in st.session_state: st.session_state.custom_gpu_tdp = None
if 'current_cpu_tdp' not in st.session_state: st.session_state.current_cpu_tdp = None
if 'current_gpu_tdp' not in st.session_state: st.session_state.current_gpu_tdp = None

if 'ig_steps_str' not in st.session_state: st.session_state.ig_steps_str = "50"
if 'ig_internal_batch_str' not in st.session_state: st.session_state.ig_internal_batch_str = "2"
if 'ig_baseline_mode' not in st.session_state: st.session_state.ig_baseline_mode = "Zeros (Black)"
if 'gs_samples_str' not in st.session_state: st.session_state.gs_samples_str = "10"
if 'gs_stdevs_str' not in st.session_state: st.session_state.gs_stdevs_str = "0.0001"
if 'gs_baseline_mode' not in st.session_state: st.session_state.gs_baseline_mode = "Zeros & Mean"
if 'occlusion_window_str' not in st.session_state: st.session_state.occlusion_window_str = "15"
if 'occlusion_stride_str' not in st.session_state: st.session_state.occlusion_stride_str = "8"
if 'occlusion_value_str' not in st.session_state: st.session_state.occlusion_value_str = "0"
if 'lime_samples_str' not in st.session_state: st.session_state.lime_samples_str = "500"
if 'lime_batch_str' not in st.session_state: st.session_state.lime_batch_str = "10"
if 'lime_segments_str' not in st.session_state: st.session_state.lime_segments_str = "50"

# --- SIDEBAR ---

# --- Restore Config Request (Must happen before any widgets are instantiated) ---
if st.session_state.get("restore_config"):
    restore_data = st.session_state.restore_config
    settings = restore_data.get("settings", {})
    environment = restore_data.get("environment", {})
    
    # Restore algorithm-specific parameter values if present
    xai_params = settings.get("xai_params", {})
    for k, v in xai_params.items():
        st.session_state[k] = v
        
    st.session_state.selected_models = settings.get("models", ["resnet50"])
    
    # Map and restore each method version individually from methods_info
    methods_map = {m.lower().replace("_", ""): m for m in xai_opts}
    restored_methods = []
    
    # We retrieve methods_info from settings or top-level restore data
    methods_info = settings.get("methods_info") or restore_data.get("methods_info")
    
    if methods_info:
        version_counts = {}
        for run in methods_info:
            base_lower = run.get("base_name", "")
            base_norm = base_lower.lower().replace("_", "").replace("-", "")
            
            # Find matching base name from available XAI options
            official_base = None
            for m_opt in xai_opts:
                if m_opt.lower().replace("_", "") == base_norm:
                    official_base = m_opt
                    break
                    
            if not official_base:
                continue
                
            version_counts[official_base] = version_counts.get(official_base, 0) + 1
            count = version_counts[official_base]
            restored_name = f"{official_base}_{count}" if count > 1 else official_base
            
            if restored_name not in restored_methods:
                restored_methods.append(restored_name)
                
            # Restore settings into session state for this restored_name
            params = run.get("params", {})
            if official_base == "Integrated_Gradients":
                st.session_state[f"ig_steps_str_{restored_name}"] = str(params.get("n_steps", 50))
                st.session_state[f"ig_internal_batch_str_{restored_name}"] = str(params.get("internal_batch_size", 2))
                st.session_state[f"ig_baseline_mode_{restored_name}"] = params.get("baseline_mode", "Zeros (Black)")
            elif official_base == "Gradient_Shap":
                st.session_state[f"gs_samples_str_{restored_name}"] = str(params.get("n_samples", 10))
                st.session_state[f"gs_stdevs_str_{restored_name}"] = str(params.get("stdevs", 0.0001))
                st.session_state[f"gs_baseline_mode_{restored_name}"] = params.get("baseline_mode", "Zeros & Mean")
            elif official_base == "Occlusion":
                sliding = params.get("sliding_window_shapes", (3, 15, 15))
                strds = params.get("strides", (3, 8, 8))
                st.session_state[f"occlusion_window_str_{restored_name}"] = str(sliding[1] if len(sliding) > 1 else 15)
                st.session_state[f"occlusion_stride_str_{restored_name}"] = str(strds[1] if len(strds) > 1 else 8)
                st.session_state[f"occlusion_value_str_{restored_name}"] = str(params.get("occlude_color", 0))
            elif official_base == "Lime":
                st.session_state[f"lime_samples_str_{restored_name}"] = str(params.get("n_samples", 500))
                st.session_state[f"lime_batch_str_{restored_name}"] = str(params.get("perturbations_per_eval", 10))
                st.session_state[f"lime_segments_str_{restored_name}"] = str(params.get("n_segments", 50))
    else:
        # Fallback parsing for legacy configs that only have expanded current_batch_methods
        legacy_methods = settings.get("methods", [])
        for base in ["Integrated_Gradients", "Gradient_Shap", "Occlusion", "Lime"]:
            has_v1 = False
            for m in legacy_methods:
                if m.startswith(base):
                    if m == base or re.match(rf"^{base}_1(?:_\d+)?$", m):
                        has_v1 = True
                        break
            if not has_v1:
                for m in legacy_methods:
                    if m.startswith(base) and not any(m.startswith(f"{base}_{v}") for v in [2, 3, 4, 5]):
                        has_v1 = True
                        break
            if has_v1:
                if base in methods_map.values():
                    restored_methods.append(base)
                
            for v in range(2, 6):
                for m in legacy_methods:
                    if m.startswith(f"{base}_{v}"):
                        restored_name = f"{base}_{v}"
                        if restored_name not in restored_methods:
                            restored_methods.append(restored_name)
                        break
                        
        for m in legacy_methods:
            base_m = re.sub(r'_\d+$', '', m)
            if base_m not in ["Integrated_Gradients", "Gradient_Shap", "Occlusion", "Lime"]:
                base_norm = base_m.lower().replace("_", "")
                if base_norm in methods_map:
                    restored_name = methods_map[base_norm]
                    if restored_name not in restored_methods:
                        restored_methods.append(restored_name)
                    
    st.session_state.selected_methods = restored_methods if restored_methods else ["Saliency", "Integrated_Gradients"]
    
    raw_sizes = settings.get("input_sizes") or settings.get("input_size_str") or [224]
    if isinstance(raw_sizes, list) and len(raw_sizes) > 0:
        st.session_state.selected_input_sizes = [str(s) for s in raw_sizes]
    elif isinstance(raw_sizes, (str, int)):
        if isinstance(raw_sizes, str):
            st.session_state.selected_input_sizes = [s.strip() for s in raw_sizes.split(",") if s.strip()]
        else:
            st.session_state.selected_input_sizes = [str(raw_sizes)]
    else:
        st.session_state.selected_input_sizes = ["224"]
    
    # Ensure options contains all selected ones
    for s in st.session_state.selected_input_sizes:
        if s not in st.session_state.input_sizes_options:
            st.session_state.input_sizes_options.append(s)
            
    st.session_state.input_size_str = ", ".join(st.session_state.selected_input_sizes)
    st.session_state.selected_repeats = settings.get("repeat_count", 5)
    st.session_state.selected_warmups = settings.get("warmup_runs", 3)
    st.session_state.selected_memory_runs = settings.get("memory_runs", 1)
    st.session_state.selected_run_order = settings.get("run_order", "Balanced")
    st.session_state.selected_random_seed = settings.get("random_seed", 42)
    st.session_state.enable_quality_metrics = settings.get("enable_quality_metrics", False)
    st.session_state.selected_quality_metrics = settings.get("selected_quality_metrics") or ["Gini Index (Sparsity)", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)"]
    
    env_dev = environment.get("selected_device")
    if env_dev:
        env_dev_str = str(env_dev).lower()
        if "cuda" in env_dev_str:
            st.session_state.selected_device_mode = "GPU (CUDA)"
        elif "mps" in env_dev_str:
            st.session_state.selected_device_mode = "GPU (MPS)"
        else:
            st.session_state.selected_device_mode = "CPU"
            
    st.session_state.custom_cpu_tdp = settings.get("custom_cpu_tdp")
    st.session_state.custom_gpu_tdp = settings.get("custom_gpu_tdp")
    st.session_state.current_cpu_tdp = settings.get("custom_cpu_tdp")
    st.session_state.current_gpu_tdp = settings.get("custom_gpu_tdp")

    # Clear the restoration request so it only runs once
    del st.session_state.restore_config
    st.session_state.config_restored_toast = True

if st.session_state.get("config_restored_toast"):
    st.toast("Configuration successfully loaded! Switched to 'Benchmark Workspace'.", icon="✅")
    st.session_state.config_restored_toast = False
    
    # Inject JavaScript to automatically switch active tab to "Benchmark Workspace" (Index 0)
    js_switch = """
    <script>
        function clickTabByIndex(index) {
            var selectors = [
                'button[role="tab"]',
                'button[data-baseweb="tab"]',
                'button[data-testid="stWidgetTab"]'
            ];
            for (var s = 0; s < selectors.length; s++) {
                var tabs = window.parent.document.querySelectorAll(selectors[s]);
                if (tabs.length > index) {
                    var tab = tabs[index];
                    if (tab.getAttribute("aria-selected") === "true") {
                        return false;
                    }
                    tab.click();
                    return true;
                }
            }
            return false;
        }
        setTimeout(function() { clickTabByIndex(0); }, 150);
    </script>
    """
    components.html(js_switch, height=0)

fragment_api = getattr(st, "fragment", getattr(st, "experimental_fragment", None))

# --- TABS WORKSPACE ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Benchmark Workspace", 
    "🗃️ Results History & Evaluation", 
    "📚 Documentation", 
    "📖 Citation"
])

with tab1:
    if st.session_state.benchmark_running or st.session_state.benchmark_ready_to_run or st.session_state.is_finished:
        render_active_run_page()
    else:
        render_configure_page()

with tab2:
    if st.session_state.benchmark_running and not st.session_state.is_finished:
        st.warning("⚙️ **Benchmark is running in the background.** This page will refresh automatically as tasks complete. We recommend staying on the **Benchmark Workspace** tab to monitor progress.")
    render_history_page()

with tab3:
    render_documentation_page()

with tab4:
    if st.session_state.benchmark_running and not st.session_state.is_finished:
        st.warning("⚡ **Benchmark is running in the background.** This page will refresh automatically as tasks complete. We recommend staying on the **Benchmark Workspace** tab to monitor progress.")
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown("If you use this benchmark in your research, papers, or projects, please cite it using the following BibTeX entry:")
    
    st.code("""@software{alimzade2025xai,
  author  = {Anar Alimzade},
  title   = {Efficiency Benchmark for XAI},
  year    = {2025},
  month   = {May},
  version = {1.0.0}
}""", language="bibtex")

# --- ENGINE ---
if st.session_state.benchmark_running and not st.session_state.is_finished and not st.session_state.get("first_run_render"):
    idx = st.session_state.run_progress_idx
    task_queue = st.session_state.task_queue
    run_img_sources = st.session_state.prepared_img_sources

    if st.session_state.last_run_batch_id != st.session_state.current_batch_id:
        st.session_state.last_run_results = []
        st.session_state.last_run_batch_id = st.session_state.current_batch_id

    if idx < len(task_queue):
        task = task_queue[idx]
        img_i = task["img_i"]
        
        src = run_img_sources[img_i]
        model_name = task["model_name"]
        target_size = task["target_size"]
        method_name = task["method_name"]
        
        target_group = get_or_create_result_group(st.session_state.last_run_results, img_i, run_img_sources)
        model_label = f"{model_name} ({target_size}px)"
        model_entry = next((m for m in target_group["models"] if m.get("model_label") == model_label), None)
        
        if not model_entry:
            s_dir = sm.get_task_path(st.session_state.current_batch_id, img_i + 1, f"{model_name}_{target_size}")
            if hasattr(src, 'getbuffer'):
                tp = os.path.join(s_dir, "input_image.jpg")
                with open(tp, "wb") as f:
                    f.write(src.getbuffer())
                fs = tp
            else:
                fs = src
            
            # Check if prediction and original resolution are already computed (e.g. from a resumed run)
            pred_val = "Unknown"
            orig_res_val = "Unknown"
            config_path = os.path.normpath(os.path.join(s_dir, "config.json"))
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        t_cfg = json.load(f)
                    pred_val = t_cfg.get("prediction", "Unknown")
                    orig_res_val = t_cfg.get("original_resolution", "Unknown")
                except Exception:
                    pass
                    
            model_entry = {"model": model_name, "model_label": model_label, "input_size": target_size, "results": [], "session_dir": s_dir, "src_path": fs, "prediction": pred_val, "original_resolution": orig_res_val}
            target_group["models"].append(model_entry)
            
            # If the prediction is unknown, run a fast prediction-only pass to get it immediately
            if pred_val == "Unknown":
                run_benchmark_task({
                    "model_name": model_name, 
                    "image_source": model_entry["src_path"], 
                    "methods": [], 
                    "method_params": {},
                    "force_device": get_device_string(st.session_state.current_device_mode), 
                    "input_size": target_size,
                    "warmup_runs": 0,
                    "memory_runs": 0,
                    "repeat_count": 1,
                    "run_order": st.session_state.current_run_order,
                    "enable_quality_metrics": False,
                    "selected_quality_metrics": [],
                    "custom_cpu_tdp": st.session_state.get("current_cpu_tdp"),
                    "custom_gpu_tdp": st.session_state.get("current_gpu_tdp")
                }, s_dir)
                
                # Load the newly saved prediction and original resolution
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            t_cfg = json.load(f)
                        model_entry["prediction"] = t_cfg.get("prediction", "Unknown")
                        model_entry["original_resolution"] = t_cfg.get("original_resolution", "Unknown")
                    except Exception:
                        pass
                
                st.rerun()

        results = run_benchmark_task({
            "model_name": model_name, 
            "image_source": model_entry["src_path"], 
            "methods": [method_name], 
            "method_params": task.get("method_params", {}),
            "force_device": get_device_string(st.session_state.current_device_mode), 
            "input_size": target_size,
            "warmup_runs": st.session_state.current_warmups,
            "memory_runs": st.session_state.current_memory_runs,
            "repeat_count": st.session_state.current_repeats,
            "run_order": st.session_state.current_run_order,
            "enable_quality_metrics": st.session_state.current_enable_quality_metrics,
            "selected_quality_metrics": st.session_state.current_selected_quality_metrics,
            "custom_cpu_tdp": st.session_state.get("current_cpu_tdp"),
            "custom_gpu_tdp": st.session_state.get("current_gpu_tdp")
        }, model_entry["session_dir"])
        
        # Explicit composite Task ID (Image + Model + Resolution + Method) to guarantee 100% mathematical uniqueness
        task_id = f"img{img_i}_{model_name}_{target_size}px_{method_name.lower()}"
        for res in results:
            res["_task_id"] = task_id
            
        existing_tasks = {r.get("_task_id"): idx_r for idx_r, r in enumerate(model_entry["results"]) if r.get("_task_id")}
        for res in results:
            t_id = res.get("_task_id")
            if t_id and t_id in existing_tasks:
                model_entry["results"][existing_tasks[t_id]] = res
            else:
                model_entry["results"].append(res)
                
        # Read the latest prediction and original resolution from config.json and update model_entry
        config_path = os.path.normpath(os.path.join(model_entry["session_dir"], "config.json"))
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    t_cfg = json.load(f)
                model_entry["prediction"] = t_cfg.get("prediction", "Unknown")
                model_entry["original_resolution"] = t_cfg.get("original_resolution", "Unknown")
            except Exception:
                pass
                
        st.session_state.run_progress_idx += 1
        
        if st.session_state.run_progress_idx >= len(task_queue):
            st.session_state.is_finished = True
            st.session_state.benchmark_running = False
            st.session_state.completed_batch_id = st.session_state.current_batch_id
            st.session_state.total_execution_time = time.time() - st.session_state.batch_start_time
            st.session_state.batch_completed_at = timestamp_now()
            write_current_batch_results_json()
        st.rerun()

    else:
        st.session_state.is_finished = True
        st.session_state.benchmark_running = False
        st.session_state.completed_batch_id = st.session_state.current_batch_id
        if not st.session_state.batch_completed_at:
            st.session_state.batch_completed_at = timestamp_now()
        write_current_batch_results_json()
        st.rerun()


plt.close('all')
