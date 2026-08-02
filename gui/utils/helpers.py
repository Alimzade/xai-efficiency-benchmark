"""
General Utilities Module
Contains generic helper functions for timestamp formatting, device strings, parsing configurations, and app routing.
"""
import os
import json
import random
import streamlit as st
from config import PROJECT_ROOT
from datetime import datetime

from backend.benchmark_runner import get_cpu_name
fragment_api = getattr(st, 'fragment', getattr(st, 'experimental_fragment', None))


def get_device_string(mode_str):
    if "CUDA" in mode_str:
        return "cuda"
    elif "MPS" in mode_str:
        return "mps"
    return "cpu"

def build_task_queue(num_images, models, sizes, methods, run_order, seed=None):
    # Note: methods here is a list of dicts: [{"display_name": "...", "base_name": "...", "params": {...}}, ...]
    tasks = []
    if run_order == "Balanced":
        for img_i in range(num_images):
            for mod_i, model_name in enumerate(models):
                # Rotate methods instead of sizes to distribute workload variety,
                # while preserving small-to-large size ordering for fast UI feedback.
                rotation = (img_i + mod_i) % len(methods)
                ordered_methods = methods[rotation:] + methods[:rotation]
                
                for method_info in ordered_methods:
                    for target_size in sorted(sizes):
                        tasks.append({
                            "img_i": img_i,
                            "model_name": model_name,
                            "target_size": target_size,
                            "method_name": method_info["display_name"],
                            "method_base": method_info["base_name"],
                            "method_params": method_info["params"]
                        })
    else:
        for img_i in range(num_images):
            for model_name in models:
                for method_info in methods:
                    for target_size in sorted(sizes):
                        tasks.append({
                            "img_i": img_i,
                            "model_name": model_name,
                            "target_size": target_size,
                            "method_name": method_info["display_name"],
                            "method_base": method_info["base_name"],
                            "method_params": method_info["params"]
                        })

    if run_order == "Randomized":
        rng = random.Random(seed)
        rng.shuffle(tasks)
    return tasks

def sorted_result_groups(groups):
    return sorted(groups, key=lambda g: g.get("img_idx", 0))

def methods_from_result_groups(groups):
    methods = []
    for group in groups:
        for model_entry in group.get("models", []):
            for result in model_entry.get("results", []):
                method = result.get("Method")
                if method and method not in methods:
                    methods.append(method)
    return methods

def get_cpu_info(): return get_cpu_name()

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    elif seconds < 3600: return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else: return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"

def timestamp_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def display_timestamp(value):
    if not value:
        return "unknown"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        return str(value)

def format_run_timestamps(started_iso, completed_iso):
    if not started_iso:
        return ""
    try:
        dt_start = datetime.fromisoformat(started_iso)
        tz_str = dt_start.strftime("%Z")
        if not tz_str:
            offset = dt_start.utcoffset()
            if offset is not None:
                offset_hours = int(offset.total_seconds() / 3600)
                tz_str = f"UTC{'+' if offset_hours >= 0 else ''}{offset_hours}"
            else:
                tz_str = "UTC"
            
        date_start = dt_start.strftime("%b %d, %Y")
        time_start = dt_start.strftime("%H:%M:%S")
        
        if completed_iso:
            dt_end = datetime.fromisoformat(completed_iso)
            date_end = dt_end.strftime("%b %d, %Y")
            time_end = dt_end.strftime("%H:%M:%S")
            
            if date_start == date_end:
                return f"""
                <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
                    <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Date:</span>
                    <code>{date_start}</code>
                    <span style="color: rgba(148, 163, 184, 0.35); margin: 0 6px;">|</span>
                    <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Time:</span>
                    <code>{time_start}</code>
                    <span style="color: rgba(148, 163, 184, 0.35); margin: 0 3px;">&rarr;</span>
                    <code>{time_end}</code>
                    <span style="font-size: 0.76rem; color: var(--xai-muted); margin-left: 4px;">({tz_str})</span>
                </div>
                """
            else:
                return f"""
                <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
                    <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Started:</span>
                    <code>{date_start} {time_start}</code>
                    <span style="color: rgba(148, 163, 184, 0.35); margin: 0 6px;">|</span>
                    <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Completed:</span>
                    <code>{date_end} {time_end}</code>
                    <span style="font-size: 0.76rem; color: var(--xai-muted); margin-left: 4px;">({tz_str})</span>
                </div>
                """
        else:
            return f"""
            <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
                <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Started:</span>
                <code>{date_start} {time_start}</code>
                <span style="font-size: 0.76rem; color: var(--xai-muted); margin-left: 4px;">({tz_str})</span>
            </div>
            """
    except Exception:
        start_display = display_timestamp(started_iso)
        if completed_iso:
            end_display = display_timestamp(completed_iso)
            return f"""
            <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
                <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Started:</span>
                <code>{start_display}</code>
                <span style="color: rgba(148, 163, 184, 0.35); margin: 0 6px;">|</span>
                <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Completed:</span>
                <code>{end_display}</code>
            </div>
            """
        return f"""
        <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
            <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Started:</span>
            <code>{start_display}</code>
        </div>
        """

def format_run_duration(duration):
    if not duration:
        return ""
    return f"""
    <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.2rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
        <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Total Duration:</span>
        <code>{duration}</code>
    </div>
    """

def parse_input_sizes(size_str):
    try:
        sizes = [int(s.strip()) for s in size_str.split(",") if s.strip().isdigit()]
        return sizes or [224]
    except Exception:
        return [224]

@st.cache_data(show_spinner=False)
def get_batch_display_name(bid, base_dir):
    batch_meta_p = os.path.join(base_dir, bid, "batch_results.json")
    if os.path.exists(batch_meta_p):
        try:
            with open(batch_meta_p, 'r') as f:
                meta = json.load(f)
            
            # Format date & time from: Batch_YYYYMMDD_HHMMSS or Batch_YYYY-MM-DD_HH-MM-SS
            parts = bid.split("_")
            if len(parts) >= 3:
                date_part = parts[1]
                time_part = parts[2]
                
                # Format date to DD-MM-YYYY
                if len(date_part) == 8: # YYYYMMDD
                    year = date_part[:4]
                    month = date_part[4:6]
                    day = date_part[6:]
                    date_str = f"{day}-{month}-{year}"
                elif "-" in date_part: # YYYY-MM-DD
                    y_m_d = date_part.split("-")
                    if len(y_m_d) == 3:
                        date_str = f"{y_m_d[2]}-{y_m_d[1]}-{y_m_d[0]}"
                    else:
                        date_str = date_part
                else:
                    date_str = date_part
                    
                # Format time
                if len(time_part) == 6: # HHMMSS
                    time_str = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
                else:
                    time_str = time_part.replace("-", ":")
                    
                # 4 non-breaking spaces between date and time
                display_time = f"{date_str}\u00A0\u00A0\u00A0\u00A0{time_str}"
            else:
                display_time = bid
                
            results = meta.get("results", [])
            img_count = len(results)
            
            settings = meta.get("benchmark_settings", {}) or {}
            models = settings.get("models", []) or []
            methods = settings.get("methods", []) or meta.get("methods", []) or []
            resolutions = settings.get("resolutions", []) or []
            
            # Fallback if settings are empty
            if not models or not methods or not resolutions:
                scanned_models = set()
                scanned_methods = set()
                scanned_res = set()
                for g in results:
                    for m in g.get("models", []):
                        if m.get("model_name"):
                            scanned_models.add(m.get("model_name"))
                        for r in m.get("results", []):
                            if r.get("Method"):
                                scanned_methods.add(r.get("Method"))
                            if r.get("Resolution"):
                                scanned_res.add(r.get("Resolution"))
                if not models:
                    models = list(scanned_models)
                if not methods:
                    methods = list(scanned_methods)
                if not resolutions:
                    resolutions = list(scanned_res)
                
            img_lbl = f"{img_count} img" if img_count == 1 else f"{img_count} imgs"
            model_lbl = f"{len(models)} model" if len(models) == 1 else f"{len(models)} models"
            method_lbl = f"{len(methods)} method" if len(methods) == 1 else f"{len(methods)} methods"
            res_lbl = f"{len(resolutions)} res" if len(resolutions) == 1 else f"{len(resolutions)} res"
            
            # 4 non-breaking spaces before details parenthesis
            return f"{display_time}\u00A0\u00A0\u00A0\u00A0({img_lbl}, {res_lbl}, {model_lbl}, {method_lbl})"
        except Exception:
            return bid
    return bid

def rerun_app():
    try:
        st.rerun(scope="app")
    except TypeError:
        st.rerun()

def rerun_fragment():
    if fragment_api:
        st.rerun(scope="fragment")
    else:
        rerun_app()

def keep_local_images_expanded():
    st.session_state.local_images_expanded = True

def load_docs_reference():
    docs_path = os.path.join(PROJECT_ROOT, "gui", "assets", "docs_reference.json")
    if os.path.exists(docs_path):
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading documentation reference file: {e}")
    return {}
