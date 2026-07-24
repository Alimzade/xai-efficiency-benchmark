"""
State Management Module
Handles reading, writing, and synchronizing benchmark results with Streamlit session state and persistent JSON disk storage.
"""
import os
import time
import json
import streamlit as st
import pandas as pd

from gui.core import sm, PROJECT_ROOT
from gui.utils.helpers import get_device_string, build_task_queue, sorted_result_groups, timestamp_now
from gui.backend.benchmark_runner import collect_environment_metadata

def get_or_create_result_group(results, img_i, img_sources):
    img_idx = img_i + 1
    group = next((g for g in results if g.get("img_idx") == img_idx), None)
    if group is None:
        group = {"img_idx": img_idx, "models": [], "source": img_sources[img_i]}
        results.append(group)
        results.sort(key=lambda g: g.get("img_idx", 0))
    return group

def current_image_sources():
    uploaded_files = st.session_state.get("uploaded_files") or []
    url_list = [u.strip() for u in st.session_state.persisted_urls.split("\n") if u.strip()]
    
    # Auto-load local images from gui/images/ folder if it exists
    local_images = []
    images_dir = os.path.join(PROJECT_ROOT, "gui", "images")
    if os.path.exists(images_dir) and os.path.isdir(images_dir):
        try:
            active_files = st.session_state.get("active_local_filenames")
            for f in sorted(os.listdir(images_dir)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                    if active_files is None or f in active_files:
                        local_images.append(os.path.join(images_dir, f))
        except Exception as e:
            st.error(f"Error loading local images: {str(e)}")
            
    return local_images + list(uploaded_files) + url_list

def serialize_and_persist_image_sources(img_sources, batch_id, base_dir="gui/sessions"):
    persisted_sources = []
    uploaded_dir = os.path.join(base_dir, batch_id, "uploaded_images")
    
    for idx, src in enumerate(img_sources):
        if isinstance(src, str):
            persisted_sources.append(src)
        else:
            # It's a Streamlit UploadedFile or file-like object
            if not os.path.exists(uploaded_dir):
                os.makedirs(uploaded_dir, exist_ok=True)
            
            ext = "jpg"
            if hasattr(src, "name") and src.name:
                parts = src.name.rsplit(".", 1)
                if len(parts) > 1:
                    ext = parts[1].lower()
            
            filename = f"img_{idx}.{ext}"
            dest_path = os.path.normpath(os.path.join(uploaded_dir, filename))
            
            # Save the file content
            with open(dest_path, "wb") as f:
                if hasattr(src, "getbuffer"):
                    f.write(src.getbuffer())
                else:
                    f.write(src.read())
            
            persisted_sources.append(dest_path)
            
    return persisted_sources

def check_and_reset_session_after_delete(deleted_bid):
    if (st.session_state.get("current_batch_id") == deleted_bid or 
        st.session_state.get("last_run_batch_id") == deleted_bid or 
        st.session_state.get("completed_batch_id") == deleted_bid):
        st.session_state.is_finished = False
        st.session_state.benchmark_running = False
        st.session_state.benchmark_ready_to_run = False
        st.session_state.run_progress_idx = 0
        st.session_state.task_queue = []
        st.session_state.prepared_img_sources = []
        st.session_state.current_batch_id = ""
        st.session_state.batch_started_at = ""
        st.session_state.batch_completed_at = ""
        st.session_state.total_execution_time = 0
        st.session_state.batch_start_time = None
        st.session_state.last_run_results = []
        st.session_state.completed_batch_id = ""
        st.session_state.last_run_batch_id = ""
        st.session_state.current_batch_methods = []
        st.session_state.current_batch_models = []
        st.session_state.current_batch_sizes = []

def write_current_batch_results_json():
    batch_id = st.session_state.current_batch_id
    if not batch_id:
        return
    clean_results = []
    for g in sorted_result_groups(st.session_state.last_run_results):
        cg = g.copy()
        if hasattr(cg["source"], 'name'): cg["source"] = cg["source"].name
        clean_results.append(cg)
        
    results_path = os.path.join(sm.base_dir, batch_id, "batch_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "results": clean_results, 
            "methods": st.session_state.current_batch_methods,
            "benchmark_settings": {
                "warmup_runs": st.session_state.current_warmups,
                "memory_runs": st.session_state.current_memory_runs,
                "repeat_count": st.session_state.current_repeats,
                "enable_quality_metrics": st.session_state.current_enable_quality_metrics,
                "selected_quality_metrics": st.session_state.current_selected_quality_metrics,
                "run_order": st.session_state.current_run_order,
                "task_count": len(st.session_state.task_queue),
                "models": st.session_state.current_batch_models,
                "input_sizes": st.session_state.current_batch_sizes,
                "methods": st.session_state.current_batch_methods,
                "selected_methods": st.session_state.sh_methods if st.session_state.get("sh_methods") else st.session_state.selected_methods,
                "xai_params": st.session_state.get("sh_xai_params", {})
            },
            "environment": collect_environment_metadata(
                get_device_string(st.session_state.current_device_mode),
                custom_cpu_tdp=st.session_state.get("current_cpu_tdp"),
                custom_gpu_tdp=st.session_state.get("current_gpu_tdp")
            ),
            "started_at": st.session_state.batch_started_at,
            "completed_at": st.session_state.batch_completed_at,
            "total_execution_time": st.session_state.total_execution_time,
            "methods_info": st.session_state.get("current_batch_methods_info", [])
        }, f, indent=4)

def resume_batch(batch_id):
    cfg = sm.load_batch_config(batch_id)
    if not cfg:
        st.error(f"Failed to load configuration for batch {batch_id}")
        return
        
    st.session_state.current_batch_id = batch_id
    st.session_state.last_run_batch_id = batch_id
    st.session_state.current_run_order = cfg.get("run_order", "Balanced")
    st.session_state.current_batch_methods = list(cfg.get("methods", []))
    st.session_state.current_batch_methods_info = cfg.get("methods_info", [])
    st.session_state.current_batch_models = list(cfg.get("models", []))
    st.session_state.current_batch_sizes = list(cfg.get("input_sizes", []))
    st.session_state.current_device_mode = cfg.get("device_mode", "CPU")
    st.session_state.current_warmups = cfg.get("warmup_runs", 1)
    st.session_state.current_repeats = cfg.get("repeat_count", 5)
    st.session_state.current_memory_runs = cfg.get("memory_runs", 1)
    st.session_state.current_enable_quality_metrics = cfg.get("enable_quality_metrics", False)
    st.session_state.current_selected_quality_metrics = list(cfg.get("selected_quality_metrics", []))
    st.session_state.prepared_img_sources = list(cfg.get("image_sources", []))
    st.session_state.batch_started_at = cfg.get("started_at", "")
    st.session_state.current_cpu_tdp = cfg.get("custom_cpu_tdp")
    st.session_state.current_gpu_tdp = cfg.get("custom_gpu_tdp")
    st.session_state.custom_cpu_tdp = cfg.get("custom_cpu_tdp")
    st.session_state.custom_gpu_tdp = cfg.get("custom_gpu_tdp")
    
    # Rebuild the exact same task queue using parameter-expanded methods
    expanded_methods = cfg.get("methods_info", [])
    if not expanded_methods:
        expanded_methods = [{"display_name": m, "base_name": m.lower().replace("-", "_"), "params": {}} for m in cfg.get("methods", [])]
        
    st.session_state.task_queue = build_task_queue(
        len(st.session_state.prepared_img_sources),
        st.session_state.current_batch_models,
        st.session_state.current_batch_sizes,
        expanded_methods,
        st.session_state.current_run_order,
        seed=batch_id
    )
    
    # Calculate TDP in W for the resumed session
    dev_mode_str = get_device_string(cfg.get("device_mode", "CPU"))
    res_tdp_w = cfg.get("custom_gpu_tdp") if dev_mode_str in ["cuda", "mps"] else cfg.get("custom_cpu_tdp")

    # Reload completed results from the filesystem
    st.session_state.last_run_results = []
    st.session_state.run_progress_idx = 0
    
    for idx, task in enumerate(st.session_state.task_queue):
        img_i = task["img_i"]
        model_name = task["model_name"]
        target_size = task["target_size"]
        method_name = task["method_name"]
        
        s_dir = sm.get_task_path(batch_id, img_i + 1, f"{model_name}_{target_size}")
        csv_path = os.path.normpath(os.path.join(s_dir, "results.csv"))
        config_path = os.path.normpath(os.path.join(s_dir, "config.json"))
        
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
        
        if os.path.exists(csv_path):
            try:
                df_res = pd.read_csv(csv_path)
                method_rows = df_res[df_res["Method"].str.lower() == method_name.lower()]
                if not method_rows.empty:
                    target_group = get_or_create_result_group(st.session_state.last_run_results, img_i, st.session_state.prepared_img_sources)
                    model_label = f"{model_name} ({target_size}px)"
                    model_entry = next((m for m in target_group["models"] if m.get("model_label") == model_label), None)
                    if not model_entry:
                        model_entry = {
                            "model": model_name,
                            "model_label": model_label,
                            "input_size": target_size,
                            "results": [],
                            "session_dir": s_dir,
                            "src_path": st.session_state.prepared_img_sources[img_i],
                            "prediction": pred_val,
                            "original_resolution": orig_res_val
                        }
                        target_group["models"].append(model_entry)
                    else:
                        model_entry["prediction"] = pred_val
                        model_entry["original_resolution"] = orig_res_val
                    
                    res_dicts = method_rows.to_dict(orient="records")
                    task_id = f"img{img_i}_{model_name}_{target_size}px_{method_name.lower()}"
                    for r in res_dicts:
                        r["_task_id"] = task_id
                        # Back-calculate energy consumption if missing
                        if res_tdp_w is not None:
                            try:
                                rt = r.get("Attribution Runtime (sec)")
                                if rt is None:
                                    rt = r.get("Runtime (sec)")
                                if rt is not None and ("Estimated Energy Consumption (kWh)" not in r or r["Estimated Energy Consumption (kWh)"] is None or pd.isna(r["Estimated Energy Consumption (kWh)"])):
                                    r["Estimated Energy Consumption (kWh)"] = round((float(rt) * float(res_tdp_w)) / (3600.0 * 1000.0), 8)
                            except Exception:
                                pass
                        
                    existing_tasks = {r.get("_task_id"): idx_r for idx_r, r in enumerate(model_entry["results"]) if r.get("_task_id")}
                    for r in res_dicts:
                        t_id = r.get("_task_id")
                        if t_id and t_id in existing_tasks:
                            model_entry["results"][existing_tasks[t_id]] = r
                        else:
                            model_entry["results"].append(r)
                            
                    if idx == st.session_state.run_progress_idx:
                        st.session_state.run_progress_idx = idx + 1
                    continue
            except Exception:
                pass
        break
        
    st.session_state.is_finished = (st.session_state.run_progress_idx >= len(st.session_state.task_queue))
    if st.session_state.is_finished:
        st.session_state.benchmark_running = False
        st.session_state.benchmark_ready_to_run = False
        st.session_state.completed_batch_id = batch_id
        if not st.session_state.batch_completed_at:
            st.session_state.batch_completed_at = timestamp_now()
        write_current_batch_results_json()
    else:
        st.session_state.benchmark_running = True
        st.session_state.benchmark_ready_to_run = True
        
    st.session_state.stop_requested = False
    st.session_state.current_page = "Active Run"
    st.session_state.batch_start_time = time.time()
