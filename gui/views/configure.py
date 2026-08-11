"""
Configuration Page Module
Renders the setup UI where users configure models, datasets, methods, and hardware parameters before starting a benchmark.
"""
import os
import sys
import re
import torch
import platform
import streamlit as st
from dbgpu import GPUDatabase
from PIL import Image

from config import sm, PROJECT_ROOT, model_opts, fixed_size_models, min_input_size, region_based_methods
from backend.benchmark_runner import find_cpu_tdp
from components.media import get_base64, render_image_preview_gallery
from utils.loader import expand_xai_methods_with_params, assign_numbered_suffixes
from utils.helpers import build_task_queue, get_cpu_info, parse_input_sizes, rerun_app, keep_local_images_expanded
from utils.state import current_image_sources, serialize_and_persist_image_sources, check_and_reset_session_after_delete, resume_batch

def render_configure_page():
    # Callback to handle dynamically adding a custom size without raising StreamlitAPIException
    def add_custom_size_callback():
        val = st.session_state.get("new_custom_size_input", "").strip()
        if val:
            if val.isdigit():
                val_int = int(val)
                if val_int >= min_input_size:
                    if val not in st.session_state.input_sizes_options:
                        st.session_state.input_sizes_options.append(val)
                    if val not in st.session_state.selected_input_sizes:
                        st.session_state.selected_input_sizes.append(val)
                    
                    # Sort them immediately
                    st.session_state.input_sizes_options = sorted(
                        list(set(st.session_state.input_sizes_options)),
                        key=lambda x: int(x) if x.isdigit() else 0
                    )
                    st.session_state.selected_input_sizes = sorted(
                        list(set(st.session_state.selected_input_sizes)),
                        key=lambda x: int(x) if x.isdigit() else 0
                    )
                else:
                    st.warning(f"⚠️ Input size must be at least {min_input_size}px.")
            else:
                st.warning("⚠️ Please enter a valid number.")
        st.session_state.new_custom_size_input = ""

    # Detect and initialize auto-loaded folder images state (Safe from widget lock here!)
    local_all = []
    images_dir = os.path.join(PROJECT_ROOT, "images")
    if os.path.exists(images_dir) and os.path.isdir(images_dir):
        try:
            for f in sorted(os.listdir(images_dir)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                    local_all.append(f)
            
            if local_all:
                if 'active_local_filenames' not in st.session_state:
                    st.session_state.active_local_filenames = local_all.copy()
                    st.session_state.previous_local_all = local_all.copy()
                else:
                    prev_all = st.session_state.get("previous_local_all", [])
                    new_files = [f for f in local_all if f not in prev_all]
                    if new_files:
                        st.session_state.active_local_filenames = list(set(st.session_state.active_local_filenames + new_files))
                    st.session_state.previous_local_all = local_all.copy()
                
                # Filter out deleted files
                st.session_state.active_local_filenames = [
                    f for f in st.session_state.active_local_filenames if f in local_all
                ]
        except Exception as e:
            st.error(f"Error initializing local images: {str(e)}")

    if st.session_state.get("stop_requested") and st.session_state.get("sh_models"):
        st.session_state.selected_models = st.session_state.sh_models
        st.session_state.selected_methods = st.session_state.sh_methods
        st.session_state.input_size_str = st.session_state.sh_input_size_str
        st.session_state.selected_input_sizes = [s.strip() for s in st.session_state.sh_input_size_str.split(",") if s.strip()]
        for s in st.session_state.selected_input_sizes:
            if s not in st.session_state.input_sizes_options:
                st.session_state.input_sizes_options.append(s)
        st.session_state.selected_repeats = st.session_state.sh_repeats
        st.session_state.selected_warmups = st.session_state.sh_warmups
        st.session_state.selected_run_order = st.session_state.sh_run_order
        st.session_state.selected_device_mode = st.session_state.sh_device_mode
        st.session_state.stop_requested = False

    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)

    incomplete_batches = sm.list_incomplete_batches()
    if incomplete_batches:
        st.markdown("### ⚡ Interrupted Benchmarks")
        st.warning("The following benchmarks were interrupted (e.g. due to system sleep/restart). You can resume them from where they left off.")
        
        for b in incomplete_batches:
            col_b1, col_b2, col_b3 = st.columns([3, 1, 1])
            with col_b1:
                st.markdown(f"**Batch:** `{b['id']}` ({b['created']})  \n`{b['info']}`")
            with col_b2:
                if st.button("Resume", key=f"resume_{b['id']}", use_container_width=True):
                    resume_batch(b['id'])
                    rerun_app()
            with col_b3:
                if st.button("Delete", key=f"del_inc_{b['id']}", use_container_width=True):
                    sm.delete_batch(b['id'])
                    check_and_reset_session_after_delete(b['id'])
                    st.success(f"Deleted {b['id']}")
                    rerun_app()
        st.markdown('<div style="margin-top: 1.5rem; margin-bottom: 1.5rem; border-top: 1px solid rgba(148, 163, 184, 0.2); padding-top: 0.5rem;"></div>', unsafe_allow_html=True)

    # Row 1
    main_left, main_right = st.columns([2, 1])
    row1_left, row1_right = main_left, main_right
    with row1_left:
        selected_models_widget = st.multiselect(
            "Model Architectures",
            model_opts,
            key="selected_models",
        )
    with row1_right:
        col_w, col_m = st.columns(2)
        with col_w:
            st.number_input(
                "Warmup runs",
                min_value=0,
                max_value=20,
                key="selected_warmups",
                step=1,
                help="Untimed runs before measurement. Useful for CUDA/model warmup.",
            )
        with col_m:
            st.number_input(
                "Memory runs",
                min_value=0,
                max_value=100,
                key="selected_memory_runs",
                step=1,
                help="Dedicated memory measurement runs (default 1). Set to 0 to skip memory profiling entirely.",
            )

    # Row 2 (Inputs Row - Top)
    row2_top_left, row2_top_right = main_left, main_right
    with row2_top_left:
        fixed_models = [m for m in selected_models_widget if m in fixed_size_models]
        if fixed_models:
            st.multiselect(
                "Input Sizes (px)",
                options=["224"],
                default=["224"],
                disabled=True,
                help="Only applicable to CNN-based architectures."
            )
            st.session_state.input_size_str = "224"
            if len(fixed_models) == 1:
                st.caption(f"⚠️ *Fixed-size architecture selected (Locked to 224px):* **`{fixed_models[0]}`**")
            else:
                formatted_models = ", ".join([f"**`{m}`**" for m in fixed_models])
                st.caption(f"⚠️ *Fixed-size architectures selected (Locked to 224px):* {formatted_models}")
        else:
            # Sort selected input sizes numerically so bubble tags are always in order
            if st.session_state.get("selected_input_sizes"):
                st.session_state.selected_input_sizes = sorted(
                    list(set(st.session_state.selected_input_sizes)),
                    key=lambda x: int(x) if x.isdigit() else 0
                )
                
            col_sel, col_add = st.columns([3, 1])
            with col_sel:
                st.multiselect(
                    "Input Sizes (px)",
                    options=st.session_state.input_sizes_options,
                    key="selected_input_sizes",
                    help="Only applicable to CNN-based architectures."
                )
            with col_add:
                st.text_input(
                    "Add size (px)",
                    key="new_custom_size_input",
                    placeholder="e.g. 416",
                    help="Type a custom size and press Enter to add it as a bubble tag.",
                    autocomplete="off",
                    on_change=add_custom_size_callback,
                )
            
            st.session_state.input_size_str = ", ".join(st.session_state.selected_input_sizes)

    with row2_top_right:
        col_r, col_o = st.columns(2)
        with col_r:
            st.number_input(
                "Measured repeats",
                min_value=1,
                max_value=1000,
                key="selected_repeats",
                step=1,
                help="Timed attribution repeats per image/model/size/method. Use 30-100 for stronger size studies when methods are fast enough.",
            )
        with col_o:
            if st.session_state.get("selected_run_order") == "Randomized":
                col_o1, col_o2 = st.columns([2.2, 1])
                with col_o1:
                    st.selectbox(
                        "Task Order",
                        options=["Balanced", "Sequential", "Randomized"],
                        key="selected_run_order",
                        help="Balanced: Rotates XAI methods. Sequential: Groups by model/size. Randomized: fully shuffled."
                    )
                with col_o2:
                    st.number_input(
                        "Seed",
                        min_value=0,
                        max_value=999999,
                        value=int(st.session_state.get("selected_random_seed", 42)),
                        key="selected_random_seed",
                        step=1,
                        help="Random seed for pseudo-random number generator (random.Random). Guarantees 100% reproducible task shuffling across benchmark runs.",
                        label_visibility="visible",
                    )
            else:
                st.selectbox(
                    "Task Order",
                    options=["Balanced", "Sequential", "Randomized"],
                    key="selected_run_order",
                    help="Balanced: Rotates XAI methods. Sequential: Groups by model/size. Randomized: fully shuffled."
                )

    # Row 2 (Inputs Row - Bottom)
    row2_mid_left, row2_mid_right = main_left, main_right
    with row2_mid_left:
        # Dynamically build options for XAI Methods.
        # This allows users to select the same method multiple times (e.g. Integrated_Gradients, Integrated_Gradients_2)
        base_xai_opts = [
            "Saliency",
            "Integrated_Gradients",
            "Guided_Backprop",
            "Input_X_Gradient",
            "Gradient_Shap",
            "DeepLift",
            "DeepLift_Shap",
            "Grad_CAM",
            "Occlusion",
            "LIME",
        ]
        
        selected_methods = st.session_state.get("selected_methods", ["Saliency", "Integrated_Gradients"])
        
        dynamic_xai_opts = list(base_xai_opts)
        parameterized_bases = ["Integrated_Gradients", "Gradient_Shap", "Occlusion", "LIME"]
        for base in base_xai_opts:
            if base not in parameterized_bases:
                continue
            pattern = re.compile(rf"^{base}(?:_(\d+))?$")
            versions = []
            for m in selected_methods:
                match = pattern.match(m)
                if match:
                    val = match.group(1)
                    versions.append(int(val) if val else 1)
            
            if versions:
                next_version = max(versions) + 1
                dynamic_xai_opts.append(f"{base}_{next_version}")
                for v in sorted(versions):
                    name = f"{base}_{v}" if v > 1 else base
                    if name not in dynamic_xai_opts:
                        dynamic_xai_opts.append(name)

        st.multiselect(
            "XAI Methods",
            dynamic_xai_opts,
            key="selected_methods",
        )
        
        # Compact XAI Method Parameters rendered right under the selector in the same column
        modifiable_selected = []
        for m in st.session_state.selected_methods:
            base_name = re.sub(r'_\d+$', '', m)
            if base_name in ["Integrated_Gradients", "Gradient_Shap", "Occlusion", "LIME"]:
                modifiable_selected.append(m)
        
        if modifiable_selected:
            # Force defaults back into session state if they were deleted during streamlit's widget cleanup
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
            
            # Pre-fill method-specific parameter defaults
            for method in modifiable_selected:
                base = re.sub(r'_\d+$', '', method)
                if base == "Integrated_Gradients":
                    if f"ig_steps_str_{method}" not in st.session_state: st.session_state[f"ig_steps_str_{method}"] = st.session_state.get("ig_steps_str", "50")
                    if f"ig_internal_batch_str_{method}" not in st.session_state: st.session_state[f"ig_internal_batch_str_{method}"] = st.session_state.get("ig_internal_batch_str", "2")
                    if f"ig_baseline_mode_{method}" not in st.session_state: st.session_state[f"ig_baseline_mode_{method}"] = st.session_state.get("ig_baseline_mode", "Zeros (Black)")
                elif base == "Gradient_Shap":
                    if f"gs_samples_str_{method}" not in st.session_state: st.session_state[f"gs_samples_str_{method}"] = st.session_state.get("gs_samples_str", "10")
                    if f"gs_stdevs_str_{method}" not in st.session_state: st.session_state[f"gs_stdevs_str_{method}"] = st.session_state.get("gs_stdevs_str", "0.0001")
                    if f"gs_baseline_mode_{method}" not in st.session_state: st.session_state[f"gs_baseline_mode_{method}"] = st.session_state.get("gs_baseline_mode", "Zeros & Mean")
                elif base == "Occlusion":
                    if f"occlusion_window_str_{method}" not in st.session_state: st.session_state[f"occlusion_window_str_{method}"] = st.session_state.get("occlusion_window_str", "15")
                    if f"occlusion_stride_str_{method}" not in st.session_state: st.session_state[f"occlusion_stride_str_{method}"] = st.session_state.get("occlusion_stride_str", "8")
                    if f"occlusion_value_str_{method}" not in st.session_state: st.session_state[f"occlusion_value_str_{method}"] = st.session_state.get("occlusion_value_str", "0")
                elif base == "LIME":
                    if f"lime_samples_str_{method}" not in st.session_state: st.session_state[f"lime_samples_str_{method}"] = st.session_state.get("lime_samples_str", "500")
                    if f"lime_batch_str_{method}" not in st.session_state: st.session_state[f"lime_batch_str_{method}"] = st.session_state.get("lime_batch_str", "10")
                    if f"lime_segments_str_{method}" not in st.session_state: st.session_state[f"lime_segments_str_{method}"] = st.session_state.get("lime_segments_str", "50")

    with row2_mid_right:
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        device_options = ["CPU"]
        if has_cuda:
            device_options.insert(0, "GPU (CUDA)")
        elif has_mps:
            device_options.insert(0, "GPU (MPS)")

        if st.session_state.selected_device_mode not in device_options:
            st.session_state.selected_device_mode = device_options[0]
        st.markdown('<label class="custom-hw-label" data-testid="stWidgetLabel">Hardware Device</label>', unsafe_allow_html=True)
        def set_device_mode(mode):
            st.session_state.selected_device_mode = mode

        _btn_cols = st.columns(len(device_options))
        for _i, _opt in enumerate(device_options):
            with _btn_cols[_i]:
                _is_selected = st.session_state.selected_device_mode == _opt
                st.button(
                    _opt,
                    key=f"device_btn_{_opt}",
                    use_container_width=True,
                    type="primary" if _is_selected else "secondary",
                    disabled=_is_selected,
                    on_click=set_device_mode,
                    args=(_opt,)
                )

    # Determine current hardware details
    cpu_name = get_cpu_info()
    gpu_desc = ""
    if torch.cuda.is_available():
        gpu_desc = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        gpu_desc = "Apple Silicon (MPS)"
    else:
        gpu_desc = "Generic GPU"

    detected_cpu_tdp = None
    matched_cpu = None
    try:
        detected_cpu_tdp, matched_cpu = find_cpu_tdp(cpu_name)
    except Exception:
        pass

    detected_gpu_tdp = None
    matched_gpu = None
    if torch.cuda.is_available():
        try:
            db = GPUDatabase.default()
            spec = db.search(gpu_desc)
            if spec and hasattr(spec, "thermal_design_power_w") and spec.thermal_design_power_w:
                detected_gpu_tdp = int(spec.thermal_design_power_w)
                matched_gpu = getattr(spec, "name", None)
        except Exception:
            pass

    # Initialize session state keys if None using auto-detected values or fallbacks
    if st.session_state.custom_cpu_tdp is None:
        st.session_state.custom_cpu_tdp = int(detected_cpu_tdp) if detected_cpu_tdp else 65
    if st.session_state.custom_gpu_tdp is None:
        st.session_state.custom_gpu_tdp = int(detected_gpu_tdp) if detected_gpu_tdp else 250

    # Adjust TDP input UI depending on selected mode
    is_gpu = "GPU" in st.session_state.selected_device_mode
    
    tdp_line = ""
    if is_gpu:
        active_tdp = st.session_state.custom_gpu_tdp
        if active_tdp:
            if detected_gpu_tdp and active_tdp != detected_gpu_tdp:
                tdp_line = f"<p>• <strong>GPU TDP</strong>: {active_tdp} W (User Override, auto-detected: {detected_gpu_tdp} W)</p>"
            elif detected_gpu_tdp:
                if matched_gpu and matched_gpu.lower().strip() != gpu_desc.lower().strip():
                    tdp_line = f"<p>• <strong>GPU TDP</strong>: {active_tdp} W (matched to: {matched_gpu})</p>"
                else:
                    tdp_line = f"<p>• <strong>GPU TDP</strong>: {active_tdp} W</p>"
            else:
                tdp_line = f"<p>• <strong>GPU TDP</strong>: {active_tdp} W (User Specified)</p>"
        status_text = f"🟢 {gpu_desc}"
    else:
        active_tdp = st.session_state.custom_cpu_tdp
        if active_tdp:
            if detected_cpu_tdp and active_tdp != detected_cpu_tdp:
                tdp_line = f"<p>• <strong>CPU TDP</strong>: {active_tdp} W (User Override, auto-detected: {detected_cpu_tdp} W)</p>"
            elif detected_cpu_tdp:
                if matched_cpu and matched_cpu.lower().strip() != cpu_name.lower().strip():
                    tdp_line = f"<p>• <strong>CPU TDP</strong>: {active_tdp} W (matched to: {matched_cpu})</p>"
                else:
                    tdp_line = f"<p>• <strong>CPU TDP</strong>: {active_tdp} W</p>"
            else:
                tdp_line = f"<p>• <strong>CPU TDP</strong>: {active_tdp} W (User Specified)</p>"
        status_text = f"💻 {cpu_name}"
    
    # Build environment details lines dynamically to prevent blank lines
    details_lines = [
        f"<p>• <strong>OS/Platform</strong>: {platform.platform()}</p>",
        f"<p>• <strong>Python Version</strong>: {sys.version.split()[0]}</p>",
        f"<p>• <strong>PyTorch Version</strong>: {torch.__version__}</p>"
    ]
    
    if torch.cuda.is_available():
        details_lines.append(f"<p>• <strong>CUDA Version</strong>: {torch.version.cuda}</p>")
    if tdp_line:
        details_lines.append(tdp_line)
        
    details_content = "\n".join(details_lines)

    # Bottom Layout Columns (isolating expansion on the left and right)
    bottom_left, bottom_right = main_left, main_right
    with bottom_left:
        if modifiable_selected:
            with st.expander("⚙️ Algorithm Parameter Tuning"):
                st.markdown('<div style="font-size: 0.8em; color: var(--xai-muted); margin-bottom: 12px;">Customize algorithm inputs below. Add same method again or use comma-separated values (e.g. <b>50, 100</b>) to test multiple parameter variations.</div>', unsafe_allow_html=True)
                
                for idx_m, method in enumerate(modifiable_selected):
                    base_method = re.sub(r'_\d+$', '', method)
                    
                    if idx_m > 0:
                        st.markdown("<hr style='margin: 4px 0 16px 0; border: 0; border-top: 1px solid rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
                        
                    st.markdown(f"<div style='font-size: 0.85em; font-weight: 600; color: var(--xai-text); margin-top: 0px; margin-bottom: 8px;'>{method.replace('_', ' ')}</div>", unsafe_allow_html=True)
                    
                    from config import method_configs
                    if base_method in method_configs:
                        params_dict = method_configs[base_method]
                        cols = st.columns(len(params_dict))
                        for col, (param_key, param_info) in zip(cols, params_dict.items()):
                            with col:
                                if param_info["type"] == "select":
                                    opts = param_info["choices"]
                                    default_val = param_info["default"]
                                    saved_val = st.session_state.get(f"{param_key}_{method}", st.session_state.get(param_key, default_val))
                                    idx = opts.index(saved_val) if saved_val in opts else 0
                                    st.selectbox(
                                        param_info["label"],
                                        options=opts,
                                        index=idx,
                                        key=f"{param_key}_{method}",
                                        help=param_info.get("help", "")
                                    )
                                else:
                                    default_val = str(param_info["default"])
                                    saved_val = st.session_state.get(f"{param_key}_{method}", st.session_state.get(param_key, default_val))
                                    st.text_input(
                                        param_info["label"],
                                        value=str(saved_val),
                                        key=f"{param_key}_{method}",
                                        help=param_info.get("help", "")
                                    )

        # Measurement Details collapsible below parameters inside left column
        meas_details_html = """
        <style>
            .meas-details-container {
                border: 1px solid var(--xai-border) !important;
                border-radius: 8px !important;
                overflow: hidden !important;
                background: rgba(11, 18, 27, 0.25) !important;
                transition: border-color 0.2s ease !important;
                margin-top: 0px !important;
                margin-bottom: 0px !important;
            }
            .meas-details-container:hover {
                border-color: rgba(96, 165, 250, 0.3) !important;
            }
            .meas-details-summary {
                display: flex !important;
                justify-content: space-between !important;
                align-items: center !important;
                padding: 0.6rem 1rem !important;
                font-weight: 600 !important;
                font-size: 0.9rem !important;
                color: var(--xai-text) !important;
                cursor: pointer !important;
                list-style: none !important;
                background: rgba(255, 255, 255, 0.04) !important;
                transition: background-color 0.2s ease !important;
            }
            .meas-details-summary:hover {
                background-color: rgba(255, 255, 255, 0.08) !important;
            }
            .meas-details-summary::-webkit-details-marker {
                display: none !important;
            }
            .meas-details-container[open] .meas-chevron {
                transform: rotate(180deg) !important;
            }
            .meas-details-content {
                padding: 1rem !important;
                border-top: 1px solid var(--xai-border) !important;
                font-size: 0.875rem !important;
                color: var(--xai-muted) !important;
                background: rgba(11, 18, 27, 0.45) !important;
            }
            div[class*="st-key-custom_cpu_tdp"],
            div[class*="st-key-custom_gpu_tdp"],
            div[class*="st-key-custom_cpu_tdp"] [data-testid="stNumberInput"],
            div[class*="st-key-custom_gpu_tdp"] [data-testid="stNumberInput"] {
                margin-top: 0px !important;
                padding-top: 0px !important;
            }
        </style>
        <details class="meas-details-container" open>
            <summary class="meas-details-summary">
                <span>Measurement Details</span>
                <span class="meas-chevron" style="transition: transform 0.2s; font-size: 0.75rem; display: inline-block;">▼</span>
            </summary>
            <div class="meas-details-content">
                <ul class="nice-bullets" style="margin-top: 0px; margin-bottom: 0px; padding-left: 20px; list-style-type: disc;">
                    <li style="margin-bottom: 8px;"><b>Warmups & Repeats</b>: Warmups are excluded. Repeats measure device duration and report stats (median, mean, std).</li>
                    <li style="margin-bottom: 8px;"><b>Task Ordering</b>: Controls execution sequence of configurations to eliminate systematic execution position bias.</li>
                    <li style="margin-bottom: 8px;"><b>Timing & Memory Isolation</b>: Memory profiling runs in a dedicated iteration to keep timing runs clean of overhead.</li>
                    <li style="margin-bottom: 0px;"><b>Energy Estimation</b>: TDP is used as a proxy scaling factor: <code>Energy (kWh) = Runtime (sec) * TDP (W) / (3600 * 1000)</code>.</li>
                </ul>
            </div>
        </details>
        """
        st.markdown("\n".join([line.strip() for line in meas_details_html.split("\n") if line.strip()]), unsafe_allow_html=True)

    with bottom_right:
        status_html = f"""
        <style>
            .hw-status-summary:hover,
            .meas-details-summary:hover {{
                background-color: rgba(255, 255, 255, 0.05) !important;
            }}
            .hw-status-container {{
                border: 1px solid var(--xai-border) !important;
                border-radius: 8px !important;
                overflow: hidden !important;
                background: linear-gradient(135deg, rgba(96, 165, 250, 0.14) 0%, rgba(45, 212, 191, 0.08) 100%),
                            repeating-linear-gradient(-45deg, rgba(255, 255, 255, 0.015) 0px, rgba(255, 255, 255, 0.015) 2px, transparent 2px, transparent 10px) !important;
                transition: border-color 0.2s ease !important;
                margin-top: 0px !important;
                margin-bottom: 16px !important;
            }}
            .hw-status-container:hover {{
                border-color: rgba(45, 212, 191, 0.35) !important;
            }}
            .hw-status-summary {{
                display: flex !important;
                justify-content: space-between !important;
                align-items: center !important;
                padding: 0.5rem 1rem !important;
                font-weight: 400 !important;
                font-size: 0.875rem !important;
                color: var(--xai-text) !important;
                cursor: pointer !important;
                list-style: none !important;
            }}
            .hw-status-summary::-webkit-details-marker {{
                display: none !important;
            }}
            .hw-status-container[open] .hw-chevron {{
                transform: rotate(180deg) !important;
            }}
            .hw-status-details {{
                padding: 1rem !important;
                border-top: 1px solid var(--xai-border) !important;
                font-size: 0.875rem !important;
                color: var(--xai-muted) !important;
                background: rgba(11, 18, 27, 0.45) !important;
            }}
            .hw-status-details p {{
                margin: 0 0 6px 0 !important;
                line-height: 1.6 !important;
            }}
            .hw-status-details p:last-child {{
                margin-bottom: 0 !important;
            }}
        </style>
        <details class="hw-status-container">
            <summary class="hw-status-summary">
                <span>{status_text}</span>
                <span class="hw-chevron" style="transition: transform 0.2s; font-size: 0.75rem; display: inline-block;">▼</span>
            </summary>
            <div class="hw-status-details">
                <p><strong>Hardware & Environment Details:</strong></p>
                {details_content}
            </div>
        </details>
        """
        clean_html = "\n".join([line.strip() for line in status_html.split("\n") if line.strip()])
        st.markdown(clean_html, unsafe_allow_html=True)

        if is_gpu:
            with st.container():
                st.number_input(
                    "Device TDP (Watts)",
                    min_value=1,
                    max_value=1500,
                    value=int(st.session_state.custom_gpu_tdp),
                    key="custom_gpu_tdp",
                    help="Adjust the Thermal Design Power (TDP) in Watts for your active GPU. This affects energy estimation logic."
                )
                if detected_gpu_tdp:
                    if matched_gpu and matched_gpu.lower().strip() != gpu_desc.lower().strip():
                        text = f"Auto-detected: {detected_gpu_tdp}W (matched to: <i>{matched_gpu}</i>)"
                    else:
                        text = f"Auto-detected: {detected_gpu_tdp}W"
                else:
                    text = "Could not auto-detect GPU TDP. Using default fallback. Adjust if incorrect."
                st.markdown(f'<div class="tdp-caption-wrapper" style="font-size: 0.8rem; color: var(--xai-muted);">{text}</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.number_input(
                    "Device TDP (Watts)",
                    min_value=1,
                    max_value=1000,
                    value=int(st.session_state.custom_cpu_tdp),
                    key="custom_cpu_tdp",
                    help="Adjust the Thermal Design Power (TDP) in Watts for your active CPU. This affects energy estimation logic."
                )
                if detected_cpu_tdp:
                    if matched_cpu and matched_cpu.lower().strip() != cpu_name.lower().strip():
                        text = f"Auto-detected: {detected_cpu_tdp}W (matched to: <i>{matched_cpu}</i>)"
                    else:
                        text = f"Auto-detected: {detected_cpu_tdp}W"
                else:
                    text = "Could not auto-detect CPU TDP. Using default fallback. Adjust if incorrect."
                st.markdown(f'<div class="tdp-caption-wrapper" style="font-size: 0.8rem; color: var(--xai-muted);">{text}</div>', unsafe_allow_html=True)

    # Row 5: Quality Metrics (Post-Processing)
    st.divider()
    all_quality_opts = ["Gini Index (Sparsity)", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)", "Infidelity (Perturbation Faithfulness)"]
    default_quality_opts = ["Gini Index (Sparsity)", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)"]

    def on_quality_toggle_change():
        if st.session_state.get("enable_quality_metrics") and not st.session_state.get("selected_quality_metrics"):
            st.session_state.selected_quality_metrics = list(default_quality_opts)

    is_quality_enabled = st.toggle(
        "Explanation Quality Evaluation",
        key="enable_quality_metrics",
        on_change=on_quality_toggle_change,
        help="Evaluates explanation quality (e.g. Gini Index / Sparsity) in post-processing outside the timing and memory benchmarking clock."
    )

    if is_quality_enabled and not st.session_state.selected_quality_metrics:
        st.session_state.selected_quality_metrics = list(default_quality_opts)

    st.multiselect(
        "Select Quality Metrics",
        all_quality_opts,
        key="selected_quality_metrics",
        disabled=not is_quality_enabled,
        help="Post-hoc quality metrics evaluated outside the timing clock. Gini Index (sparsity), Deletion AUC (faithfulness upon removal), Insertion AUC (faithfulness upon addition), Sensitivity (Max) (worst-case sensitivity), and Infidelity (perturbation robustness)."
    )

    if is_quality_enabled and any("Infidelity" in m for m in st.session_state.selected_quality_metrics):
        selected_m = st.session_state.get("selected_methods", [])
        active_region_methods = [m for m in selected_m if any(rm.lower() in m.lower() for rm in region_based_methods)]
        if active_region_methods:
            formatted_rm = ", ".join([f"**{m}**" for m in active_region_methods])
            st.warning(f"⚠️ **Caution on Infidelity**: Infidelity is computed in pixel space using fine-grained Gaussian perturbations ($\\delta^T A(x)$). For your selected region-based methods ({formatted_rm}), this metric is mathematically ill-suited and will produce distorted, invalid scores because coarse patch attributions cannot accurately track high-frequency pixel noise.")
        else:
            st.caption("ℹ️ *Infidelity is active and will evaluate pixel-space perturbation faithfulness for your selected gradient attribution methods.*")

    st.divider()

    col_input, col_spacer, col_preview = st.columns([2.5, 0.2, 0.8])
    with col_input:
        uploaded_files = st.file_uploader(
            "Drag and drop images",
            type=["jpg", "jpeg", "png", "webp", "gif"],
            accept_multiple_files=True,
            key="uploaded_files"
        )
        
        # Auto-loaded folder images expander
        if local_all:
            active_count = len(st.session_state.active_local_filenames)
            expanded_val = st.session_state.get("local_images_expanded", False)
            with st.expander(f"Auto-Loaded Images ({active_count}/{len(local_all)})", expanded=expanded_val):
                st.session_state.local_images_expanded = False  # Reset state after rendering
                st.markdown('<div style="font-size: 0.85em; color: var(--xai-muted); margin-bottom: 8px;">Images loaded automatically from <code>gui/images/</code>. Select or remove images to customize the benchmark workload.</div>', unsafe_allow_html=True)
                st.multiselect(
                    "Included Images",
                    options=local_all,
                    default=local_all,
                    key="active_local_filenames",
                    label_visibility="collapsed",
                    on_change=keep_local_images_expanded
                )
                
                st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
                del_col_select, del_col_btn = st.columns([3.2, 0.8])
                with del_col_select:
                    file_to_del = st.selectbox(
                        "Delete file from disk:",
                        options=["-- Choose file to delete from disk --"] + local_all,
                        key="local_file_to_delete",
                        label_visibility="collapsed",
                        on_change=keep_local_images_expanded
                    )
                with del_col_btn:
                    if file_to_del != "-- Choose file to delete from disk --":
                        if st.button("🗑️ Delete", key="delete_local_file_btn", help="Permanently delete this file from your local disk.", use_container_width=True):
                            path = os.path.join(images_dir, file_to_del)
                            if os.path.exists(path):
                                try:
                                    os.remove(path)
                                    st.session_state.local_images_expanded = True  # Keep expander open on rerun!
                                    st.toast(f"Deleted {file_to_del} from disk.", icon="🗑️")
                                    if file_to_del in st.session_state.active_local_filenames:
                                        st.session_state.active_local_filenames.remove(file_to_del)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                    else:
                        st.button("🗑️ Delete", disabled=True, use_container_width=True)
        
        with st.expander("Paste Image URLs", expanded=False):
            st.text_area(
                "Image URLs",
                height=100,
                help="Please put each URL on a new line, without commas or other separators.",
                key="persisted_urls"
            )
    with col_preview:
        render_image_preview_gallery(current_image_sources())

    # Dynamic CSS to replace default file list icons with actual image thumbnails
    if uploaded_files:
        css_rules = []
        for idx, file in enumerate(uploaded_files):
            try:
                file.seek(0)
                img = Image.open(file)
                img.thumbnail((60, 60))
                b64 = get_base64(img)
                css_rules.append(f"""
                /* Target file icon specifically (exclude deletion cross inside button) */
                [data-testid="stFileUploader"] [role="list"] > *:nth-child({idx + 1}) svg:not(button svg),
                [data-testid="stFileUploader"] li:nth-child({idx + 1}) svg:not(button svg),
                .stFileUploaderFile:nth-child({idx + 1}) svg:not(button svg),
                .stFileUploaderFile:nth-of-type({idx + 1}) svg:not(button svg) {{
                    display: none !important;
                }}
                
                /* Prepend the actual image thumbnail */
                [data-testid="stFileUploader"] [role="list"] > *:nth-child({idx + 1})::before,
                [data-testid="stFileUploader"] li:nth-child({idx + 1})::before,
                .stFileUploaderFile:nth-child({idx + 1})::before,
                .stFileUploaderFile:nth-of-type({idx + 1})::before {{
                    content: "";
                    display: inline-block;
                    width: 28px;
                    height: 28px;
                    border-radius: 4px;
                    margin-right: 8px;
                    background-image: url('data:image/png;base64,{b64}');
                    background-size: cover;
                    background-position: center;
                    flex-shrink: 0;
                    align-self: center;
                    border: 1px solid rgba(255, 255, 255, 0.15);
                }}
                """)
            except Exception:
                pass
        if css_rules:
            st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)
    st.divider()
    
    selected_sizes = [224] if any(m in ["vit-b-16", "swin-t"] for m in st.session_state.selected_models) else parse_input_sizes(st.session_state.input_size_str)
    img_sources = current_image_sources()
    planned_image_count = len(st.session_state.prepared_img_sources) if st.session_state.benchmark_ready_to_run else len(img_sources)
    
    st.markdown(f"""
        <div class="run-summary-bar">
            <span class="run-summary-item">Models <strong>{len(st.session_state.selected_models)}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Size variations <strong>{len(selected_sizes)}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Methods <strong>{len(st.session_state.selected_methods)}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Images <strong>{planned_image_count}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Repeats/config <strong>{st.session_state.selected_repeats}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Warmups/config <strong>{st.session_state.selected_warmups}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Memory runs/config <strong>{st.session_state.selected_memory_runs}</strong></span>
        </div>
        <div class="run-summary-action-spacer"></div>
        """, unsafe_allow_html=True)

    if st.button("Start Multi-Model Benchmark ⚡", type="primary", use_container_width=True):
        st.session_state.last_run_results = []
        st.session_state.last_run_batch_id = ""
        st.session_state.completed_batch_id = ""
        st.session_state.completion_notice_batch_id = ""
        st.session_state.is_finished = False
        st.session_state.total_execution_time = 0
        st.session_state.batch_started_at = ""
        st.session_state.batch_completed_at = ""
        st.session_state.task_queue = []
        st.session_state.prepared_img_sources = []
        st.session_state.current_batch_methods = []
        st.session_state.current_batch_models = []
        st.session_state.current_batch_sizes = []
        
        if not img_sources or not st.session_state.selected_models or not st.session_state.selected_methods:
            st.error("Select at least one image, model, input size, and XAI method.")
        elif any(s < 32 for s in selected_sizes):
            st.error("Cannot start benchmark: All input sizes must be at least 32px to prevent model architecture crashes.")
        else:
            st.session_state.sh_models = list(st.session_state.selected_models)
            st.session_state.sh_methods = list(st.session_state.selected_methods)
            st.session_state.sh_input_size_str = st.session_state.input_size_str
            st.session_state.sh_repeats = st.session_state.selected_repeats
            st.session_state.sh_warmups = st.session_state.selected_warmups
            st.session_state.sh_memory_runs = st.session_state.selected_memory_runs
            st.session_state.sh_run_order = st.session_state.selected_run_order
            st.session_state.sh_random_seed = st.session_state.selected_random_seed
            st.session_state.sh_device_mode = st.session_state.selected_device_mode
            st.session_state.sh_xai_params = {k: v for k, v in st.session_state.items() if any(prefix in k for prefix in ["ig_", "gs_", "occlusion_", "lime_"]) and any(suffix in k for suffix in ["_str", "_mode"])}

            batch_id = sm.start_batch()
            st.session_state.current_batch_id = batch_id
            st.session_state.last_run_batch_id = batch_id
            st.session_state.current_run_order = st.session_state.selected_run_order
            st.session_state.current_random_seed = st.session_state.selected_random_seed
            
            # Expand methods with parameter variations
            expanded_methods = expand_xai_methods_with_params(st.session_state.selected_methods)
            st.session_state.current_batch_methods_info = expanded_methods
            display_methods = [m["display_name"] for m in expanded_methods]
            st.session_state.current_batch_methods = display_methods
            
            st.session_state.current_batch_models = assign_numbered_suffixes(list(st.session_state.selected_models))
            st.session_state.current_batch_sizes = list(selected_sizes)
            st.session_state.current_device_mode = st.session_state.selected_device_mode
            st.session_state.current_warmups = st.session_state.selected_warmups
            st.session_state.current_repeats = st.session_state.selected_repeats
            st.session_state.current_memory_runs = st.session_state.selected_memory_runs
            st.session_state.current_enable_quality_metrics = st.session_state.enable_quality_metrics
            st.session_state.current_selected_quality_metrics = list(st.session_state.get('selected_quality_metrics', ["Gini Index (Sparsity)"]))
            st.session_state.current_cpu_tdp = st.session_state.get("custom_cpu_tdp")
            st.session_state.current_gpu_tdp = st.session_state.get("custom_gpu_tdp")
            
            # Persist and serialize image sources to survive app restarts and system sleep/shuts
            persisted_imgs = serialize_and_persist_image_sources(img_sources, batch_id, sm.base_dir)
            st.session_state.prepared_img_sources = persisted_imgs
            
            seed_to_use = st.session_state.selected_random_seed if st.session_state.selected_run_order == "Randomized" else None
            st.session_state.task_queue = build_task_queue(
                len(persisted_imgs),
                st.session_state.current_batch_models,
                st.session_state.current_batch_sizes,
                expanded_methods,
                st.session_state.selected_run_order,
                seed=seed_to_use
            )
            
            # Save batch configuration for recovery
            batch_config = {
                "batch_id": batch_id,
                "warmup_runs": st.session_state.current_warmups,
                "memory_runs": st.session_state.current_memory_runs,
                "repeat_count": st.session_state.current_repeats,
                "enable_quality_metrics": st.session_state.current_enable_quality_metrics,
                "selected_quality_metrics": st.session_state.current_selected_quality_metrics,
                "run_order": st.session_state.current_run_order,
                "random_seed": st.session_state.current_random_seed,
                "models": st.session_state.current_batch_models,
                "input_sizes": st.session_state.current_batch_sizes,
                "methods": display_methods,
                "methods_info": expanded_methods,
                "device_mode": st.session_state.current_device_mode,
                "image_sources": persisted_imgs,
                "started_at": "",
                "custom_cpu_tdp": st.session_state.current_cpu_tdp,
                "custom_gpu_tdp": st.session_state.current_gpu_tdp
            }
            sm.save_batch_config(batch_id, batch_config)
            
            st.session_state.batch_start_time = None
            st.session_state.total_execution_time = 0
            st.session_state.stop_requested = False
            st.session_state.run_progress_idx = 0
            st.session_state.benchmark_ready_to_run = True
            st.session_state.current_page = "Active Run"
            rerun_app()
