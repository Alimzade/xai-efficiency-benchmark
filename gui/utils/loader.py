"""
Data Loader Module
Provides utility functions for parsing user inputs and expanding parameter configurations (e.g. LIME samples, Occlusion strides) into distinct XAI methods.
"""

import re
import streamlit as st


def parse_comma_sep_ints(val_str, default_val):
    try:
        vals = [int(x.strip()) for x in val_str.split(",") if x.strip().isdigit()]
        return vals if vals else [default_val]
    except Exception:
        return [default_val]

def parse_comma_sep_floats(val_str, default_val):
    try:
        vals = [float(x.strip()) for x in val_str.split(",") if x.strip()]
        return vals if vals else [default_val]
    except Exception:
        return [default_val]


def expand_xai_methods_with_params(selected_methods):
    def safe_get(key, default):
        try:
            return st.session_state.get(key, default)
        except Exception:
            return default

    expanded_methods = []
    for method in selected_methods:
        base_method = re.sub(r'_\d+$', '', method)
        
        if base_method == "Lime":
            samples_list = parse_comma_sep_ints(safe_get(f"lime_samples_str_{method}", "50"), 50)
            segments_list = parse_comma_sep_ints(safe_get(f"lime_segments_str_{method}", "50"), 50)
            for n_samples in samples_list:
                for n_segments in segments_list:
                    expanded_methods.append({
                        "method": method,
                        "base_name": base_method,
                        "params": {
                            "n_samples": n_samples,
                            "batch_size": 10,
                            "n_segments": n_segments
                        }
                    })
        elif base_method == "Occlusion":
            window_list = parse_comma_sep_ints(safe_get(f"occlusion_window_str_{method}", "15"), 15)
            stride_list = parse_comma_sep_ints(safe_get(f"occlusion_stride_str_{method}", "8"), 8)
            for window in window_list:
                for stride in stride_list:
                    expanded_methods.append({
                        "method": method,
                        "base_name": base_method,
                        "params": {
                            "window_shapes": (3, window, window),
                            "strides": (3, stride, stride)
                        }
                    })
        elif base_method == "Integrated_Gradients":
            steps_list = parse_comma_sep_ints(safe_get(f"ig_steps_str_{method}", "50"), 50)
            for steps in steps_list:
                expanded_methods.append({
                    "method": method,
                    "base_name": base_method,
                    "params": {
                        "n_steps": steps
                    }
                })
        elif base_method == "Gradient_Shap":
            samples_list = parse_comma_sep_ints(safe_get(f"gs_samples_str_{method}", "5"), 5)
            stdevs_list = parse_comma_sep_floats(safe_get(f"gs_stdevs_str_{method}", "0.1"), 0.1)
            for n_samples in samples_list:
                for stdevs in stdevs_list:
                    expanded_methods.append({
                        "method": method,
                        "base_name": base_method,
                        "params": {
                            "n_samples": n_samples,
                            "stdevs": stdevs
                        }
                    })
        else:
            expanded_methods.append({
                "method": method,
                "base_name": base_method,
                "params": {}
            })
    return assign_display_name_suffixes(expanded_methods)

def assign_display_name_suffixes(expanded_methods):
    counts = {}
    for m in expanded_methods:
        name = m["method"]
        counts[name] = counts.get(name, 0) + 1
        
    seen = {}
    for m in expanded_methods:
        name = m["method"]
        if counts[name] > 1:
            seen[name] = seen.get(name, 0) + 1
            m["display_name"] = f"{name} ({seen[name]})"
        else:
            m["display_name"] = name
    return expanded_methods

def assign_numbered_suffixes(items_list):
    counts = {}
    for item in items_list:
        counts[item] = counts.get(item, 0) + 1
        
    seen = {}
    result = []
    for item in items_list:
        if counts[item] > 1:
            seen[item] = seen.get(item, 0) + 1
            result.append(f"{item} ({seen[item]})")
        else:
            result.append(item)
    return result
