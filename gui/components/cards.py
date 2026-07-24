"""
UI Cards Module
Contains reusable Streamlit UI widget rendering functions, such as environment summaries, parameter mapping tables, and detailed result cards.
"""
import os
import time
import math
import re
import colorsys
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import streamlit.components.v1 as components

from gui.core import sm
from gui.analysis.pareto import compute_pareto_ranking
from gui.analysis.metrics import image_size_summary, method_detail_summary
from gui.utils.processing import ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL, ATTR_MEMORY_COL, LEGACY_MEMORY_COL, metric_col, normalize_metric_columns, presentation_df
from gui.components.tables import style_dataframe
from gui.components.media import get_image_thumbnail_base64
from gui.components.plots import plot_runtime_memory_scatter, plot_pareto_scatter, plot_method_runtime_log, plot_method_memory, plot_model_comparison_grouped, plot_model_memory_comparison_grouped, plot_image_size_runtime_scaling, plot_image_size_memory_scaling

def render_live_elapsed_timer(start_time):
    start_ms = int((start_time or time.time()) * 1000)
    components.html(f"""
        <div id="elapsed-badge" style="
            font-family: monospace;
            font-size: 1.2em;
            font-weight: 700;
            color: #6b7280;
            text-align: right;
            white-space: nowrap;
            padding-top: 2px;
        ">0.0s</div>
        <script>
            const startMs = {start_ms};
            const badge = document.getElementById("elapsed-badge");
            function formatElapsed(seconds) {{
                if (seconds < 60) return seconds.toFixed(1) + "s";
                if (seconds < 3600) return Math.floor(seconds / 60) + "m " + Math.floor(seconds % 60) + "s";
                return Math.floor(seconds / 3600) + "h " + Math.floor((seconds % 3600) / 60) + "m";
            }}
            function tick() {{
                badge.textContent = formatElapsed((Date.now() - startMs) / 1000);
            }}
            tick();
            setInterval(tick, 500);
        </script>
    """, height=34)

def render_analytics_sections(fdf, result_groups):
    """Render adaptive, collapsible analytics sections based on which dimensions vary."""
    fdf = normalize_metric_columns(fdf)
    runtime_col = metric_col(fdf, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    memory_col = metric_col(fdf, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    
    n_methods = fdf["Method"].nunique()
    n_models = fdf["Model"].nunique()
    n_resolutions = fdf["Resolution"].nunique() if "Resolution" in fdf.columns else 1

    # ── Shared workload metric card helper ────────────────────────────────
    def _stat_card(label, value, is_result=False, suffix=""):
        if is_result:
            bg, border = "rgba(16,185,129,0.08)", "1.5px solid rgba(16,185,129,0.45)"
            lbl_clr, val_clr = "#6ee7b7", "#34d399"
            suffix_html = f'<div style="font-size:0.62rem;color:#6ee7b7;opacity:0.7;margin-top:2px;">{suffix}</div>' if suffix else ""
        else:
            bg, border = "rgba(99,102,241,0.07)", "1.5px solid rgba(99,102,241,0.35)"
            lbl_clr, val_clr = "#a5b4fc", "inherit"
            suffix_html = ""
        return (
            f'<div style="flex:1;background:{bg};border:{border};border-radius:8px;'
            f'padding:8px 12px 6px 12px;text-align:center;min-width:0;">'
            f'<div style="font-size:0.68rem;color:{lbl_clr};letter-spacing:0.07em;'
            f'text-transform:uppercase;margin-bottom:2px;">{label}</div>'
            f'<div style="font-size:1.35rem;font-weight:700;line-height:1.1;color:{val_clr};">{value}</div>'
            + suffix_html + '</div>'
        )

    def _sep(symbol):
        clr = "#34d399" if symbol == "=" else "#6366f1"
        return (
            f'<span style="font-size:1.2rem;font-weight:500;color:{clr};'
            f'flex-shrink:0;padding:0 1px;opacity:0.8;">{symbol}</span>'
        )

    def _cards_row(*items):
        """items: list of (label, value) or (label, value, is_result) or (label, value, is_result, suffix) or (sep_symbol,)"""
        html = '<div style="display:flex;align-items:center;gap:6px;width:100%;margin-bottom:12px;">'
        for item in items:
            if len(item) == 1:
                html += _sep(item[0])
            else:
                label, value = item[0], item[1]
                is_result = item[2] if len(item) > 2 else False
                suffix = item[3] if len(item) > 3 else ""
                html += _stat_card(label, value, is_result, suffix)
        html += "</div>"
        return html
    # ─────────────────────────────────────────────────────────────────────

    # If nothing varies, the per-image results table is sufficient
    if n_methods <= 1 and n_models <= 1 and n_resolutions <= 1:
        return
    
    with st.expander("📊 Performance Analysis", expanded=True):
        # 1. Configuration Averages Table at the very top
        group_cols = ["Method", "Model", "Resolution"]
        n_images = int(fdf["Image Index"].nunique()) if "Image Index" in fdf.columns else 1
        std_runtime_src = "Attribution Runtime Std (sec)" if (n_images == 1 and "Attribution Runtime Std (sec)" in fdf.columns) else runtime_col
        std_runtime_func = "mean" if n_images == 1 else "std"

        std_mem_src = "Attribution Memory Std (MB)" if (n_images == 1 and "Attribution Memory Std (MB)" in fdf.columns) else memory_col
        std_mem_func = "mean" if n_images == 1 else "std"

        agg_dict_config = {
            "Mean Attribution Runtime (sec)": (runtime_col, "mean"),
            "Attribution Runtime Std (sec)": (std_runtime_src, std_runtime_func),
        }
        if "Estimated Energy Consumption (kWh)" in fdf.columns and fdf["Estimated Energy Consumption (kWh)"].notna().any():
            agg_dict_config["Mean Estimated Energy Consumption (kWh)"] = ("Estimated Energy Consumption (kWh)", "mean")
        agg_dict_config["Mean Peak Attribution Memory (MB)"] = (memory_col, "mean")
        agg_dict_config["Peak Memory Std (MB)"] = (std_mem_src, std_mem_func)
        if "Gini Index" in fdf.columns and fdf["Gini Index"].notna().any():
            agg_dict_config["Mean Gini Index"] = ("Gini Index", "mean")
        if "Deletion AUC" in fdf.columns and fdf["Deletion AUC"].notna().any():
            agg_dict_config["Mean Deletion AUC"] = ("Deletion AUC", "mean")
        if "Insertion AUC" in fdf.columns and fdf["Insertion AUC"].notna().any():
            agg_dict_config["Mean Insertion AUC"] = ("Insertion AUC", "mean")
        if "Sensitivity (Max)" in fdf.columns and fdf["Sensitivity (Max)"].notna().any():
            agg_dict_config["Mean Sensitivity (Max)"] = ("Sensitivity (Max)", "mean")
        if "Infidelity" in fdf.columns and fdf["Infidelity"].notna().any():
            agg_dict_config["Mean Infidelity"] = ("Infidelity", "mean")
        agg_dict_config["Samples"] = (runtime_col, "count")
        summary_df = fdf.groupby(group_cols).agg(**agg_dict_config).reset_index()
        with st.expander("📋 Configuration Averages", expanded=True):
            _c_images  = int(fdf["Image Index"].nunique()) if "Image Index" in fdf.columns else 1
            _c_repeats = int(fdf["Measured Runs"].iloc[0]) if "Measured Runs" in fdf.columns else 1
            _c_config  = _c_images * _c_repeats
            _c_total   = n_models * n_methods * n_resolutions * _c_images * _c_repeats
            st.html(_cards_row(
                ("Images", _c_images),
                ("×",),
                ("Repeats", _c_repeats),
                ("=",),
                ("Runs per Config", _c_config, True),
            ))
            st.table(style_dataframe(summary_df))
        
            # 2. Side-by-side Configuration Charts of matching height (12, 7)
            cs1, cs2 = st.columns(2)
            with cs1:
                # Architecture & Resolution Efficiency Chart (Log Scale with values)
                fdf_plot = fdf.copy()
                fdf_plot["Model_Size"] = fdf_plot["Model"] + " (" + fdf_plot["Resolution"] + ")"
                
                # Build custom palette: different base color per model, shades per resolution (larger is darker)
                
                def parse_res_num(res_str):
                    match = re.search(r'\d+', str(res_str))
                    return int(match.group()) if match else 0
                
                base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                unique_models = sorted(fdf_plot["Model"].unique())
                model_to_base = {model: base_colors[idx % len(base_colors)] for idx, model in enumerate(unique_models)}
                
                custom_palette = {}
                hue_order = []
                for model in unique_models:
                    model_sizes = fdf_plot[fdf_plot["Model"] == model]["Model_Size"].unique()
                    sorted_sizes = sorted(model_sizes, key=parse_res_num)
                    hue_order.extend(sorted_sizes)
                    V = len(sorted_sizes)
                    base_color = model_to_base[model]
                    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                    
                    for i, ms in enumerate(sorted_sizes):
                        mult = 1.35 - (i / (V - 1)) * 0.7 if V > 1 else 1.0
                        new_l = max(0.1, min(0.9, l * mult))
                        new_rgb = colorsys.hls_to_rgb(h, new_l, s)
                        custom_palette[ms] = mcolors.to_hex(new_rgb)
                
                # Order X-axis Methods from fastest to slowest
                methods_order = list(fdf_plot.groupby("Method")[runtime_col].mean().sort_values(ascending=True).index)
                
                fig_bar, ax_bar = plt.subplots(figsize=(12, 7))
                sns.barplot(
                    data=fdf_plot, 
                    x="Method", 
                    order=methods_order,
                    y=runtime_col, 
                    hue="Model_Size", 
                    hue_order=hue_order,
                    palette=custom_palette, 
                    ax=ax_bar, 
                    edgecolor="black", 
                    errorbar="sd"
                )
                
                # Apply log scale
                ax_bar.set_yscale("log")
                ax_bar.set_ylabel("Mean Attribution Runtime (sec, log scale)")
                
                # Put values on top of the bars dynamically based on height with a background mask
                bg_color = ax_bar.get_facecolor()
                
                # Adaptive label rotation to prevent collisions
                num_hues = len(ax_bar.containers)
                rot = 90 if num_hues > 3 else 0
                font_sz = 7 if num_hues > 4 else 8
                pad_val = 5 if rot == 90 else 3
                
                # Add top margin to prevent rotated labels clipping
                ax_bar.margins(y=0.25 if rot == 90 else 0.15)
                
                for container in ax_bar.containers:
                    labels = []
                    for rect in container:
                        height = rect.get_height()
                        if math.isnan(height) or height <= 0:
                            labels.append("")
                        elif height < 0.1:
                            labels.append(f"{height:.3f}s")
                        else:
                            labels.append(f"{height:.2f}s")
                    bar_labels = ax_bar.bar_label(container, labels=labels, padding=pad_val, fontsize=font_sz, rotation=rot)
                    for label in bar_labels:
                        label.set_bbox(dict(facecolor=bg_color, edgecolor='none', pad=0.8, alpha=0.9))
                    
                ax_bar.set_title("Architecture & Resolution Efficiency", fontsize=14, fontweight='bold')
                plt.xticks(rotation=45)
                ax_bar.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.tight_layout()
                st.pyplot(fig_bar)
                
            with cs2:
                # Runtime vs Memory Scatter Plot (matching height 12, 7)
                st.pyplot(plot_runtime_memory_scatter(fdf, figsize=(12, 7)))
            
        st.markdown('<div style="margin-top: 1.5rem; border-top: 1px solid var(--xai-border); padding-top: 1rem;"></div>', unsafe_allow_html=True)
        render_detailed_comparisons_header("comparison_section")
        
        # 3. Nested Collapsible Sections for varying dimensions
        
        # --- Method Comparison: multiple XAI methods were benchmarked ---
        if n_methods > 1:
            with st.expander("📊 XAI Method Comparison", expanded=False):
                _m_models      = int(fdf["Model"].nunique())
                _m_resolutions = int(fdf["Resolution"].nunique())
                _m_images      = int(fdf["Image Index"].nunique()) if "Image Index" in fdf.columns else 1
                _m_repeats     = int(fdf["Measured Runs"].iloc[0]) if "Measured Runs" in fdf.columns else 1
                _m_total       = _m_models * _m_resolutions * _m_images * _m_repeats
                st.html(_cards_row(
                    ("Models", _m_models),
                    ("×",),
                    ("Resolutions", _m_resolutions),
                    ("×",),
                    ("Images", _m_images),
                    ("×",),
                    ("Repeats", _m_repeats),
                    ("=",),
                    ("Runs per Method", _m_total, True),
                ))
                st.table(style_dataframe(method_detail_summary(fdf)))
                # Calculate a universal method ordering (sorted by median runtime) so both plots align perfectly
                norm_df = normalize_metric_columns(fdf)
                r_col = metric_col(norm_df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
                method_order = list(norm_df.groupby("Method")[r_col].median().sort_values(ascending=True).index)
                
                cm1, cm2 = st.columns(2)
                with cm1:
                    st.pyplot(plot_method_runtime_log(fdf, order=method_order))
                with cm2:
                    st.pyplot(plot_method_memory(fdf, order=method_order))
                    
            # --- Explanation Quality Trade-offs ---
            has_quality = any(col in fdf.columns and fdf[col].notna().any() for col in ["Deletion AUC", "Insertion AUC", "Sensitivity (Max)", "Gini Index"])
            if has_quality:
                with st.expander("⚖️ Pareto Analysis Summary", expanded=False):
                    st.markdown("Trade-off frontiers and normalized metric comparisons across XAI methods. For scatter plots, the red dashed line represents the Pareto-optimal frontier.")
                        
                    # Bottom Grid: 3 Scatter Plots
                    pq1, pq2, pq3 = st.columns(3)
                    with pq1:
                        if "Deletion AUC" in fdf.columns and fdf["Deletion AUC"].notna().any():
                            st.pyplot(plot_pareto_scatter(fdf, runtime_col, "Deletion AUC", x_lower_better=True, y_lower_better=True, title="Runtime vs Deletion AUC"))
                    with pq2:
                        if "Insertion AUC" in fdf.columns and fdf["Insertion AUC"].notna().any():
                            st.pyplot(plot_pareto_scatter(fdf, runtime_col, "Insertion AUC", x_lower_better=True, y_lower_better=False, title="Runtime vs Insertion AUC"))
                    with pq3:
                        if "Sensitivity (Max)" in fdf.columns and fdf["Sensitivity (Max)"].notna().any():
                            st.pyplot(plot_pareto_scatter(fdf, runtime_col, "Sensitivity (Max)", x_lower_better=True, y_lower_better=True, title="Runtime vs Sensitivity (Max)"))
                            
                    # Pareto Ranking Table at the bottom
                    ranking_df = compute_pareto_ranking(fdf, runtime_col)
                    if not ranking_df.empty:
                        st.markdown("<br/><b>Non-dominated Performance Summary</b>", unsafe_allow_html=True)
                        st.caption("A method is **Pareto-optimal** if no other method strictly beats it in *both* speed and quality. **Overall** indicates the number of Pareto frontiers (out of four runtime–quality comparisons) on which the method is non-dominated.")
                        st.table(style_dataframe(ranking_df))
        
        # --- Model Comparison: multiple architectures were benchmarked ---
        if n_models > 1:
            with st.expander("🏗️ Model Comparison", expanded=False):
                unique_methods     = int(fdf["Method"].nunique())
                unique_resolutions = int(fdf["Resolution"].nunique())
                unique_images      = int(fdf["Image Index"].nunique()) if "Image Index" in fdf.columns else 1
                repeats            = int(fdf["Measured Runs"].iloc[0]) if "Measured Runs" in fdf.columns else 1
                total_runs         = unique_methods * unique_resolutions * unique_images * repeats
                st.html(_cards_row(
                    ("Methods", unique_methods),
                    ("×",),
                    ("Resolutions", unique_resolutions),
                    ("×",),
                    ("Images", unique_images),
                    ("×",),
                    ("Repeats", repeats),
                    ("=",),
                    ("Runs per Model", total_runs, True),
                ))
                
                # Performance comparison table
                model_agg_dict = {
                    "Mean Attribution Runtime (sec)": (runtime_col, "mean"),
                }
                if "Estimated Energy Consumption (kWh)" in fdf.columns and fdf["Estimated Energy Consumption (kWh)"].notna().any():
                    model_agg_dict["Mean Estimated Energy Consumption (kWh)"] = ("Estimated Energy Consumption (kWh)", "mean")
                model_agg_dict["Mean Peak Attribution Memory (MB)"] = (memory_col, "mean")
                if "Gini Index" in fdf.columns and fdf["Gini Index"].notna().any():
                    model_agg_dict["Mean Gini Index"] = ("Gini Index", "mean")
                if "Deletion AUC" in fdf.columns and fdf["Deletion AUC"].notna().any():
                    model_agg_dict["Mean Deletion AUC"] = ("Deletion AUC", "mean")
                if "Insertion AUC" in fdf.columns and fdf["Insertion AUC"].notna().any():
                    model_agg_dict["Mean Insertion AUC"] = ("Insertion AUC", "mean")
                if "Sensitivity (Max)" in fdf.columns and fdf["Sensitivity (Max)"].notna().any():
                    model_agg_dict["Mean Sensitivity (Max)"] = ("Sensitivity (Max)", "mean")
                if "Infidelity" in fdf.columns and fdf["Infidelity"].notna().any():
                    model_agg_dict["Mean Infidelity"] = ("Infidelity", "mean")
                model_agg_dict["Samples"] = (runtime_col, "count")
                model_summary = fdf.groupby("Model").agg(**model_agg_dict).reset_index().sort_values("Mean Attribution Runtime (sec)")
                st.table(style_dataframe(model_summary))
                
                # Calculate universal grouping order based on runtime
                norm_df = normalize_metric_columns(fdf)
                r_col = metric_col(norm_df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
                method_order = list(norm_df.groupby("Method")[r_col].mean().sort_values(ascending=True).index)
                model_order = list(norm_df.groupby("Model")[r_col].mean().sort_values(ascending=True).index)
                
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.pyplot(plot_model_comparison_grouped(fdf, method_order=method_order, model_order=model_order))
                with mc2:
                    st.pyplot(plot_model_memory_comparison_grouped(fdf, method_order=method_order, model_order=model_order))
        
        # --- Resolution Comparison: multiple input resolutions were benchmarked ---
        if n_resolutions > 1:
            size_summary_df = image_size_summary(fdf)
            if not size_summary_df.empty:
                with st.expander("📐 Resolution Comparison", expanded=False):
                    _r_models  = int(fdf["Model"].nunique())
                    _r_methods = int(fdf["Method"].nunique())
                    _r_images  = int(fdf["Image Index"].nunique()) if "Image Index" in fdf.columns else 1
                    _r_repeats = int(fdf["Measured Runs"].iloc[0]) if "Measured Runs" in fdf.columns else 1
                    _r_total   = _r_models * _r_methods * _r_images * _r_repeats
                    st.html(_cards_row(
                        ("Models", _r_models),
                        ("×",),
                        ("Methods", _r_methods),
                        ("×",),
                        ("Images", _r_images),
                        ("×",),
                        ("Repeats", _r_repeats),
                        ("=",),
                        ("Runs per Resolution", _r_total, True),
                    ))
                    st.table(style_dataframe(size_summary_df))
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.pyplot(plot_image_size_runtime_scaling(size_summary_df))
                    with rc2:
                        st.pyplot(plot_image_size_memory_scaling(size_summary_df))

def render_environment_summary(environment):
    if not environment:
        return
        
    # Helper to clean strings and handle missing
    def clean_val(v):
        if v is None or str(v).strip() in ["", "nan", "None", ".", "unknown", "not available"]:
            return "-"
        return str(v)

    cpu_display = clean_val(environment.get("processor"))
    cpu_tdp = environment.get("cpu_tdp_w")
    matched_cpu = environment.get("matched_cpu_name")
    if cpu_tdp:
        if matched_cpu and matched_cpu.lower().strip() != cpu_display.lower().strip():
            cpu_display = f"{cpu_display} ({cpu_tdp}W TDP, matched to: {matched_cpu})"
        else:
            cpu_display = f"{cpu_display} ({cpu_tdp}W TDP)"

    env_rows = [
        ("Platform", clean_val(environment.get("platform"))),
        ("CPU", cpu_display),
    ]

    cuda_devices = environment.get("cuda_devices", [])
    selected_device = str(environment.get("selected_device", "")).lower()
    if cuda_devices:
        gpu_details = []
        for d in cuda_devices:
            name = d.get("name", "Unknown GPU")
            tdp = d.get("tdp_w")
            matched = d.get("matched_name")
            if tdp:
                if matched and matched.lower().strip() != name.lower().strip():
                    gpu_details.append(f"{name} ({tdp}W TDP, matched to: {matched})")
                else:
                    gpu_details.append(f"{name} ({tdp}W TDP)")
            else:
                gpu_details.append(name)
        gpu_names = ", ".join(gpu_details)
        env_rows.append(("GPU Device(s)", clean_val(gpu_names)))
    elif "mps" in selected_device:
        env_rows.append(("GPU Device", "Apple Silicon MPS"))

    exec_dev = clean_val(environment.get("selected_device", "cpu")).upper()
    if "CUDA" in exec_dev or "MPS" in exec_dev:
        exec_dev = f"{exec_dev} (GPU)"
    env_rows.append(("Execution Device", exec_dev))

    python_v = clean_val(environment.get("python_version"))
    torch_v = clean_val(environment.get("torch_version"))
    cuda_v = environment.get("torch_cuda_version")
    
    if cuda_v and str(cuda_v).strip().lower() not in ["", "none", "nan", "."]:
        stack_str = f"Python {python_v} | PyTorch {torch_v} (CUDA {cuda_v})"
    else:
        stack_str = f"Python {python_v} | PyTorch {torch_v}"
        
    env_rows.append(("Software Stack", stack_str))
    env_rows.append(("Git Commit", clean_val(environment.get("git_commit"))))
    
    # Build HTML table with controlled column widths and style matching Config Metadata
    table_html = """
    <table style="width:100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.88rem; color: var(--xai-text); margin-bottom: 0.5rem;">
      <thead>
        <tr style="border-bottom: 2px solid var(--xai-border); text-align: left; color: var(--xai-muted);">
          <th style="padding: 8px 10px; width: 160px; font-weight: 600;">Field</th>
          <th style="padding: 8px 10px; font-weight: 600;">Value</th>
        </tr>
      </thead>
      <tbody>
    """
    
    for field, value in env_rows:
        table_html += f"""
        <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.12);">
          <td style="padding: 8px 10px; font-weight: 500; color: var(--xai-text);">{field}</td>
          <td style="padding: 8px 10px; color: var(--xai-muted);">{value}</td>
        </tr>
        """
        
    table_html += """
      </tbody>
    </table>
    """
    
    # Clean up whitespace to prevent markdown block parsing issues
    clean_html = table_html.replace("\n", "").replace("    ", "").strip()
    st.markdown(clean_html, unsafe_allow_html=True)

def render_parameters_mapping_table(methods_info):
    if not methods_info:
        return
        
    # Filter only methods that have configured parameters
    parameterized = [m for m in methods_info if m.get("params")]
    if not parameterized:
        return
        
    st.markdown('<div style="font-weight: 600; margin-top: 15px; margin-bottom: 8px; color: var(--xai-text);">Method Parameters</div>', unsafe_allow_html=True)
    
    table_html = (
        '<table style="width:100%; border-collapse:collapse; font-size:0.85em; margin-bottom:20px; color:var(--xai-text);">'
        '<thead>'
        '<tr style="border-bottom:2px solid rgba(255,255,255,0.15); text-align:left;">'
        '<th style="padding:8px; width:25%;">Method Identifier</th>'
        '<th style="padding:8px; width:25%;">Base Algorithm</th>'
        '<th style="padding:8px; width:50%;">Configured Parameters</th>'
        '</tr>'
        '</thead>'
        '<tbody>'
    )
    for m in parameterized:
        disp_name = m["display_name"]
        base_name = m["base_name"].replace("_", " ").title()
        params = m["params"]
        
        # Format the parameters dictionary into a readable string
        formatted_params = []
        for k, v in params.items():
            k_pretty = k.replace("_", " ").title()
            if isinstance(v, list) or isinstance(v, tuple):
                # Format sliding window shapes/strides tuple cleanly
                if len(v) == 3:
                    v_str = f"({v[0]}, {v[1]}, {v[2]})"
                else:
                    v_str = str(v)
            else:
                v_str = str(v)
            formatted_params.append(f"<b>{k_pretty}</b>: {v_str}")
        
        params_str = ", ".join(formatted_params)
        table_html += (
            '<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<td style="padding:8px; font-family:monospace; font-weight:bold; color:var(--xai-text);">{disp_name}</td>'
            f'<td style="padding:8px;">{base_name}</td>'
            f'<td style="padding:8px; line-height:1.4;">{params_str}</td>'
            '</tr>'
        )
        
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

def render_configuration_summary(settings, results):
    if not results and not settings:
        return
    # Fallback to scanning results if settings is empty/missing
    if not settings:
        settings = {}
        
    models = settings.get("models", [])
    methods = settings.get("methods", [])
    sizes = settings.get("input_sizes", [])
    repeats = settings.get("repeat_count")
    warmups = settings.get("warmup_runs")
    memory_runs = settings.get("memory_runs", 1)
    
    if not models or not methods or not sizes:
        scanned_models = set()
        scanned_methods = set()
        scanned_sizes = set()
        if results:
            for g in results:
                for m in g.get("models", []):
                    if m.get("model_name"):
                        scanned_models.add(m.get("model_name"))
                    for r in m.get("results", []):
                        if r.get("Method"):
                            scanned_methods.add(r.get("Method"))
                        if "Resolution" in r and r.get("Resolution"):
                            scanned_sizes.add(str(r.get("Resolution")))
        if not models:
            models = sorted(list(scanned_models))
        if not methods:
            methods = sorted(list(scanned_methods))
        if not sizes:
            sizes = sorted(list(scanned_sizes))
            
    if results:
        img_count = len(results)
    elif settings.get("image_sources"):
        img_count = len(settings.get("image_sources"))
    else:
        img_count = len(st.session_state.get("prepared_img_sources", []))
    
    # Helper to clean strings and handle missing
    def clean_val(v):
        if v is None or str(v).strip() in ["", "nan", "None", "."]:
            return "-"
        return str(v)
        
    quality_metrics = settings.get("selected_quality_metrics") or settings.get("quality_metrics") or []
    if not quality_metrics and settings.get("enable_quality_metrics") is False:
        quality_metrics = []
    elif not quality_metrics and results:
        scanned_qm = set()
        for g in results:
            for m in g.get("models", []):
                for r in m.get("results", []):
                    for qm_name in ["Gini Index", "Deletion AUC", "Insertion AUC", "Infidelity", "Sensitivity (Max)"]:
                        if qm_name in r and r.get(qm_name) is not None:
                            scanned_qm.add(qm_name)
        if scanned_qm:
            quality_metrics = sorted(list(scanned_qm))

    qm_count = len(quality_metrics) if quality_metrics else 0
    qm_detail = ", ".join(quality_metrics) if quality_metrics else "None (Disabled)"

    config_rows = [
        ("Images Analyzed", clean_val(img_count), "-"),
        ("Models", clean_val(len(models)), ", ".join(models) if models else "-"),
        ("Methods", clean_val(len(methods)), ", ".join(methods) if methods else "-"),
        ("Input Resolutions", clean_val(len(sizes)), ", ".join([str(s) for s in sizes]) if sizes else "-"),
        ("Repeats per Config", clean_val(repeats), "-"),
        ("Warmup Runs", clean_val(warmups), "-"),
        ("Memory Runs", clean_val(memory_runs), "-"),
        ("Quality Metrics", clean_val(qm_count), qm_detail),
    ]
    
    # Build HTML table with controlled column widths
    table_html = """
    <table style="width:100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.88rem; color: var(--xai-text); margin-bottom: 0.5rem;">
      <thead>
        <tr style="border-bottom: 2px solid var(--xai-border); text-align: left; color: var(--xai-muted);">
          <th style="padding: 8px 10px; width: 160px; font-weight: 600;">Parameter</th>
          <th style="padding: 8px 10px; width: 80px; font-weight: 600;">Count</th>
          <th style="padding: 8px 10px; font-weight: 600;">Detail</th>
        </tr>
      </thead>
      <tbody>
    """
    
    for param, count, detail in config_rows:
        table_html += f"""
        <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.12);">
          <td style="padding: 8px 10px; font-weight: 500; color: var(--xai-text);">{param}</td>
          <td style="padding: 8px 10px; color: var(--xai-muted);">{count}</td>
          <td style="padding: 8px 10px; color: var(--xai-muted);">{detail}</td>
        </tr>
        """
        
    table_html += """
      </tbody>
    </table>
    """
    
    # Clean up whitespace to prevent the markdown parser from treating indented HTML as code
    clean_html = table_html.replace("\n", "").replace("    ", "").strip()
    st.markdown(clean_html, unsafe_allow_html=True)

def render_detailed_results_header(key_prefix, title_text="Detailed Per-Image Results"):
    components.html(f"""
        <style>
            body {{
                margin: 0;
                background: transparent;
                display: flex;
                align-items: center;
                justify-content: space-between;
                height: 100%;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .xai-title {{
                font-size: 1.2rem;
                font-weight: 600;
                color: #e5edf6;
                white-space: nowrap;
                line-height: 1;
                display: flex;
                align-items: center;
            }}
            .xai-expander-controls {{
                display: flex;
                gap: 8px;
                align-items: center;
            }}
            .xai-expander-controls button {{
                border: 1px solid rgba(148, 163, 184, 0.26);
                border-radius: 8px;
                background: linear-gradient(180deg, rgba(96, 165, 250, 0.16), rgba(45, 212, 191, 0.1));
                color: #e5edf6;
                cursor: pointer;
                font: 600 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                padding: 0.35rem 0.7rem;
                white-space: nowrap;
                transition: all 0.2s ease;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                line-height: 1;
                margin: 0;
                box-sizing: border-box;
            }}
            .xai-expander-controls button:hover {{
                border-color: rgba(45, 212, 191, 0.55);
                background: linear-gradient(180deg, rgba(96, 165, 250, 0.24), rgba(45, 212, 191, 0.18));
            }}
            .xai-expander-controls button:disabled {{
                opacity: 0.32 !important;
                cursor: not-allowed !important;
                border-color: rgba(148, 163, 184, 0.1) !important;
                background: rgba(255, 255, 255, 0.02) !important;
                color: rgba(229, 237, 246, 0.35) !important;
                pointer-events: none !important;
            }}
        </style>
        <div class="xai-title">{title_text}</div>
        <div class="xai-expander-controls">
            <button type="button" id="expand-btn">Expand All</button>
            <button type="button" id="collapse-btn">Collapse All</button>
        </div>
        <script>
            const root = window.parent.document;
            const expandBtn = document.getElementById('expand-btn');
            const collapseBtn = document.getElementById('collapse-btn');
            let isTransitioning = false;
            
            function imageResultDetails() {{
                return Array.from(root.querySelectorAll('details')).filter((details) => {{
                    // Filter out elements in hidden tabs (not currently visible on screen)
                    if (details.offsetParent === null) return false;
                    const summary = details.querySelector('summary');
                    return summary && summary.textContent.includes('Results for Image');
                }});
            }}
            
            function updateButtonStates() {{
                if (isTransitioning) return;
                const details = imageResultDetails();
                if (details.length === 0) {{
                    expandBtn.disabled = true;
                    collapseBtn.disabled = true;
                    return;
                }}
                const anyOpen = details.some(d => d.open);
                const anyClosed = details.some(d => !d.open);
                
                expandBtn.disabled = !anyClosed;
                collapseBtn.disabled = !anyOpen;
            }}
            
            function toggleAll(openState) {{
                const details = imageResultDetails();
                details.forEach(d => {{
                    const summary = d.querySelector('summary');
                    if (summary) {{
                        if ((openState && !d.open) || (!openState && d.open)) {{
                            summary.click();
                        }}
                    }}
                }});
            }}
            
            expandBtn.addEventListener('click', (e) => {{
                e.preventDefault();
                const parentWin = window.parent;
                const parentDoc = parentWin.document;
                
                // Save scroll positions of all elements in the parent document that are scrolled
                const scrollStates = [];
                const allElements = parentDoc.querySelectorAll('*');
                allElements.forEach(el => {{
                    if (el.scrollTop > 0 || el.scrollLeft > 0) {{
                        scrollStates.push({{
                            element: el,
                            scrollTop: el.scrollTop,
                            scrollLeft: el.scrollLeft
                        }});
                    }}
                }});
                const docEl = parentDoc.documentElement;
                const bodyEl = parentDoc.body;
                const winScrollX = parentWin.scrollX || docEl.scrollLeft || bodyEl.scrollLeft;
                const winScrollY = parentWin.scrollY || docEl.scrollTop || bodyEl.scrollTop;

                // Lock states immediately for transitions (expand is now disabled, collapse is enabled)
                isTransitioning = true;
                expandBtn.disabled = true;
                collapseBtn.disabled = false;

                // Expand all
                toggleAll(true);

                expandBtn.blur();
                if (parentDoc.activeElement) {{
                    parentDoc.activeElement.blur();
                }}

                // Restore all scroll positions (immediate and deferred)
                const restoreScrolls = () => {{
                    scrollStates.forEach(state => {{
                        state.element.scrollTop = state.scrollTop;
                        state.element.scrollLeft = state.scrollLeft;
                    }});
                    parentWin.scrollTo(winScrollX, winScrollY);
                }};
                restoreScrolls();
                setTimeout(restoreScrolls, 10);
                setTimeout(restoreScrolls, 50);
                
                // Release lock and re-evaluate once React animations settle
                setTimeout(() => {{
                    isTransitioning = false;
                    updateButtonStates();
                }}, 1200);
            }});
            
            collapseBtn.addEventListener('click', (e) => {{
                e.preventDefault();
                const parentWin = window.parent;
                const parentDoc = parentWin.document;
                
                // Save scroll positions of all elements in the parent document that are scrolled
                const scrollStates = [];
                const allElements = parentDoc.querySelectorAll('*');
                allElements.forEach(el => {{
                    if (el.scrollTop > 0 || el.scrollLeft > 0) {{
                        scrollStates.push({{
                            element: el,
                            scrollTop: el.scrollTop,
                            scrollLeft: el.scrollLeft
                        }});
                    }}
                }});
                const docEl = parentDoc.documentElement;
                const bodyEl = parentDoc.body;
                const winScrollX = parentWin.scrollX || docEl.scrollLeft || bodyEl.scrollLeft;
                const winScrollY = parentWin.scrollY || docEl.scrollTop || bodyEl.scrollTop;

                // Lock states immediately for transitions (collapse is now disabled, expand is enabled)
                isTransitioning = true;
                collapseBtn.disabled = true;
                expandBtn.disabled = false;

                // Collapse all
                toggleAll(false);

                collapseBtn.blur();
                if (parentDoc.activeElement) {{
                    parentDoc.activeElement.blur();
                }}

                // Restore all scroll positions (immediate and deferred)
                const restoreScrolls = () => {{
                    scrollStates.forEach(state => {{
                        state.element.scrollTop = state.scrollTop;
                        state.element.scrollLeft = state.scrollLeft;
                    }});
                    parentWin.scrollTo(winScrollX, winScrollY);
                }};
                restoreScrolls();
                setTimeout(restoreScrolls, 10);
                setTimeout(restoreScrolls, 50);
                
                // Release lock and re-evaluate once React animations settle
                setTimeout(() => {{
                    isTransitioning = false;
                    updateButtonStates();
                }}, 1200);
            }});
            
            // Listen to toggle events on the parent document (capture phase because toggle does not bubble)
            root.addEventListener('toggle', (e) => {{
                if (e.target.tagName && e.target.tagName.toLowerCase() === 'details') {{
                    // Delay state check slightly to let browser layout and React render cycles settle
                    setTimeout(updateButtonStates, 100);
                }}
            }}, true);
            
            setTimeout(updateButtonStates, 200);
            setInterval(updateButtonStates, 2000);
        </script>
    """, height=38)

def render_detailed_comparisons_header(key_prefix, title_text="Detailed Comparisons"):
    components.html(f"""
        <style>
            body {{
                margin: 0;
                background: transparent;
                display: flex;
                align-items: center;
                justify-content: space-between;
                height: 100%;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .xai-title {{
                font-size: 1.2rem;
                font-weight: 600;
                color: #e5edf6;
                white-space: nowrap;
                line-height: 1;
                display: flex;
                align-items: center;
            }}
            .xai-expander-controls {{
                display: flex;
                gap: 8px;
                align-items: center;
            }}
            .xai-expander-controls button {{
                border: 1px solid rgba(148, 163, 184, 0.26);
                border-radius: 8px;
                background: linear-gradient(180deg, rgba(96, 165, 250, 0.16), rgba(45, 212, 191, 0.1));
                color: #e5edf6;
                cursor: pointer;
                font: 600 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                padding: 0.35rem 0.7rem;
                white-space: nowrap;
                transition: all 0.2s ease;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                line-height: 1;
                margin: 0;
                box-sizing: border-box;
            }}
            .xai-expander-controls button:hover {{
                border-color: rgba(45, 212, 191, 0.55);
                background: linear-gradient(180deg, rgba(96, 165, 250, 0.24), rgba(45, 212, 191, 0.18));
            }}
            .xai-expander-controls button:disabled {{
                opacity: 0.32 !important;
                cursor: not-allowed !important;
                border-color: rgba(148, 163, 184, 0.1) !important;
                background: rgba(255, 255, 255, 0.02) !important;
                color: rgba(229, 237, 246, 0.35) !important;
                pointer-events: none !important;
            }}
        </style>
        <div class="xai-title">{title_text}</div>
        <div class="xai-expander-controls">
            <button type="button" id="expand-btn">Expand All</button>
            <button type="button" id="collapse-btn">Collapse All</button>
        </div>
        <script>
            const root = window.parent.document;
            const expandBtn = document.getElementById('expand-btn');
            const collapseBtn = document.getElementById('collapse-btn');
            let isTransitioning = false;
            
            function comparisonExpanderDetails() {{
                return Array.from(root.querySelectorAll('details')).filter((details) => {{
                    if (details.offsetParent === null) return false;
                    const summary = details.querySelector('summary');
                    return summary && summary.textContent.includes('Comparison');
                }});
            }}
            
            function updateButtonStates() {{
                if (isTransitioning) return;
                const details = comparisonExpanderDetails();
                if (details.length === 0) {{
                    expandBtn.disabled = true;
                    collapseBtn.disabled = true;
                    return;
                }}
                const anyOpen = details.some(d => d.open);
                const anyClosed = details.some(d => !d.open);
                
                expandBtn.disabled = !anyClosed;
                collapseBtn.disabled = !anyOpen;
            }}
            
            function toggleAll(openState) {{
                const details = comparisonExpanderDetails();
                details.forEach(d => {{
                    const summary = d.querySelector('summary');
                    if (summary) {{
                        if ((openState && !d.open) || (!openState && d.open)) {{
                            summary.click();
                        }}
                    }}
                }});
            }}
            
            expandBtn.addEventListener('click', (e) => {{
                e.preventDefault();
                const parentWin = window.parent;
                const parentDoc = parentWin.document;
                
                const scrollStates = [];
                const allElements = parentDoc.querySelectorAll('*');
                allElements.forEach(el => {{
                    if (el.scrollTop > 0 || el.scrollLeft > 0) {{
                        scrollStates.push({{
                            element: el,
                            scrollTop: el.scrollTop,
                            scrollLeft: el.scrollLeft
                        }});
                    }}
                }});
                const docEl = parentDoc.documentElement;
                const bodyEl = parentDoc.body;
                const winScrollX = parentWin.scrollX || docEl.scrollLeft || bodyEl.scrollLeft;
                const winScrollY = parentWin.scrollY || docEl.scrollTop || bodyEl.scrollTop;

                isTransitioning = true;
                expandBtn.disabled = true;
                collapseBtn.disabled = false;

                toggleAll(true);

                expandBtn.blur();
                if (parentDoc.activeElement) {{
                    parentDoc.activeElement.blur();
                }}

                const restoreScrolls = () => {{
                    scrollStates.forEach(state => {{
                        state.element.scrollTop = state.scrollTop;
                        state.element.scrollLeft = state.scrollLeft;
                    }});
                    parentWin.scrollTo(winScrollX, winScrollY);
                }};
                restoreScrolls();
                setTimeout(restoreScrolls, 10);
                setTimeout(restoreScrolls, 50);
                
                setTimeout(() => {{
                    isTransitioning = false;
                    updateButtonStates();
                }}, 1200);
            }});
            
            collapseBtn.addEventListener('click', (e) => {{
                e.preventDefault();
                const parentWin = window.parent;
                const parentDoc = parentWin.document;
                
                const scrollStates = [];
                const allElements = parentDoc.querySelectorAll('*');
                allElements.forEach(el => {{
                    if (el.scrollTop > 0 || el.scrollLeft > 0) {{
                        scrollStates.push({{
                            element: el,
                            scrollTop: el.scrollTop,
                            scrollLeft: el.scrollLeft
                        }});
                    }}
                }});
                const docEl = parentDoc.documentElement;
                const bodyEl = parentDoc.body;
                const winScrollX = parentWin.scrollX || docEl.scrollLeft || bodyEl.scrollLeft;
                const winScrollY = parentWin.scrollY || docEl.scrollTop || bodyEl.scrollTop;

                isTransitioning = true;
                collapseBtn.disabled = true;
                expandBtn.disabled = false;

                toggleAll(false);

                collapseBtn.blur();
                if (parentDoc.activeElement) {{
                    parentDoc.activeElement.blur();
                }}

                const restoreScrolls = () => {{
                    scrollStates.forEach(state => {{
                        state.element.scrollTop = state.scrollTop;
                        state.element.scrollLeft = state.scrollLeft;
                    }});
                    parentWin.scrollTo(winScrollX, winScrollY);
                }};
                restoreScrolls();
                setTimeout(restoreScrolls, 10);
                setTimeout(restoreScrolls, 50);
                
                setTimeout(() => {{
                    isTransitioning = false;
                    updateButtonStates();
                }}, 1200);
            }});
            
            root.addEventListener('toggle', (e) => {{
                if (e.target.tagName && e.target.tagName.toLowerCase() === 'details') {{
                    setTimeout(updateButtonStates, 100);
                }}
            }}, true);
            
            setTimeout(updateButtonStates, 200);
            setInterval(updateButtonStates, 2000);
        </script>
    """, height=38)

def render_result_group(group, selected_methods, expanded=True, key_suffix=""):
    # Find input image path to generate thumbnail
    img_path = None
    for m in group.get("models", []):
        s_dir = m.get("session_dir", "")
        # Try absolute path first
        p = os.path.join(s_dir, "input_image.jpg")
        if os.path.exists(p):
            img_path = p
            break
        # Fallback to relative path resolution under base_dir
        parts = s_dir.replace("\\", "/").split("/")
        if len(parts) >= 2:
            p_rel = os.path.join(sm.base_dir, parts[-2], parts[-1], "input_image.jpg")
            if os.path.exists(p_rel):
                img_path = p_rel
                break
                
    thumb_b64 = get_image_thumbnail_base64(img_path, size=(24, 24))
    
    if thumb_b64:
        label = f"![Thumbnail]({thumb_b64})&nbsp; **Results for Image {group['img_idx']}**"
    else:
        label = f"🖼️ **Results for Image {group['img_idx']}**"
        
    exp_key = f"res_group_exp_{group['img_idx']}_{key_suffix}" if key_suffix else None
    with st.expander(label, expanded=expanded, key=exp_key):
        # Group entries by base model architecture
        architectures = []
        for m in group["models"]:
            if m["model"] not in architectures: architectures.append(m["model"])
        
        def get_size(m):
            s = m.get("input_size")
            if s:
                try: return int(s)
                except: pass
            if m.get("results") and len(m["results"]) > 0:
                try:
                    # Extract 64 from '64 x 64'
                    res_str = str(m["results"][0].get("Resolution", ""))
                    if "x" in res_str:
                        return int(res_str.split("x")[0].strip())
                except:
                    pass
            return 0

        for arch in architectures:
            arch_models = sorted(
                [m for m in group["models"] if m["model"] == arch],
                key=get_size
            )
            sample_m = arch_models[0]
            pred_class = sample_m.get("prediction", "Unknown")
            if pred_class == "Unknown" and sample_m.get("results") and len(sample_m["results"]) > 0:
                pred_class = sample_m["results"][0].get("Prediction", "Unknown")
            
            st.markdown(f"#### Model: `{arch}`")
            st.markdown(f"<div style='margin-top: -12px; margin-bottom: 12px; font-size: 0.9rem; color: #94a3b8;'>Prediction: <strong style='color: #e5edf6;'>{pred_class}</strong></div>", unsafe_allow_html=True)
            
            # Layout: Input Image (Left) | Method Collage (Right)
            col_left, col_right = st.columns([1, 3])
            
            # 1. Show Input Image once for this Architecture
            img_path = os.path.join(sample_m["session_dir"], "input_image.jpg")
            if os.path.exists(img_path):
                # Pull original resolution from the model_entry or fallback to the first result entry
                orig_res = sample_m.get("original_resolution", "Unknown")
                if orig_res == "Unknown" and sample_m["results"]:
                    orig_res = sample_m["results"][0].get("Original Resolution", "Unknown")
                col_left.image(img_path, caption=f"Input Image ({orig_res})", use_container_width=True)
            
            # 2. Show Heatmap Thumbnails (Grid layout when 1 size, Row layout when multiple sizes)
            with col_right:
                if len(arch_models) == 1:
                    # --- Single Resolution Mode: Optimal 3 or 4 column grid based on method count ---
                    m_data = arch_models[0]
                    n_meth = len(selected_methods)
                    if n_meth % 4 == 0:
                        GRID_COLUMNS = 4
                    elif n_meth % 3 == 0:
                        GRID_COLUMNS = 3
                    else:
                        # Compare row fill efficiency (remainder): pick 3 or 4 whichever leaves a fuller final row.
                        # For n_meth = 1 or 2, this naturally selects 4, keeping image sizes at a clean 25% width.
                        GRID_COLUMNS = 3 if (n_meth % 3) > (n_meth % 4) else 4
                    
                    for idx_batch in range(0, len(selected_methods), GRID_COLUMNS):
                        method_chunk = selected_methods[idx_batch:idx_batch + GRID_COLUMNS]
                        grid_cols = st.columns(GRID_COLUMNS)
                        for i_col, method in enumerate(method_chunk):
                            res = next((r for r in m_data["results"] if r["Method"].lower() == method.lower()), None)
                            with grid_cols[i_col]:
                                st.markdown(f"**{method}**")
                                if res:
                                    h_p = os.path.join(m_data["session_dir"], "heatmaps", f"{res['Method']}.png")
                                    if os.path.exists(h_p):
                                        st.image(h_p, caption=f"{m_data['input_size']}px", use_container_width=True)
                                else:
                                    st.markdown("<div style='height: 60px; border: 1px dashed rgba(148, 163, 184, 0.32); border-radius: 8px; background: rgba(255, 255, 255, 0.035); text-align: center; padding-top: 20px; color: #94a3b8; font-size: 0.7em;'>...</div>", unsafe_allow_html=True)
                else:
                    # --- Multiple Resolutions Mode: Method per row, sizes side-by-side ---
                    for method in selected_methods:
                        st.markdown(f"**{method}**")
                        num_sizes = len(arch_models)
                        cols_to_make = max(num_sizes, 4) 
                        size_cols = st.columns(cols_to_make)
                        
                        for i, m_data in enumerate(arch_models):
                            res = next((r for r in m_data["results"] if r["Method"].lower() == method.lower()), None)
                            with size_cols[i]:
                                if res:
                                    h_p = os.path.join(m_data["session_dir"], "heatmaps", f"{res['Method']}.png")
                                    if os.path.exists(h_p):
                                        st.image(h_p, caption=f"{m_data['input_size']}px", use_container_width=True)
                                else:
                                    st.markdown("<div style='height: 60px; border: 1px dashed rgba(148, 163, 184, 0.32); border-radius: 8px; background: rgba(255, 255, 255, 0.035); text-align: center; padding-top: 20px; color: #94a3b8; font-size: 0.7em;'>...</div>", unsafe_allow_html=True)
            
            # 3. Consolidated Table for all sizes of this Architecture
            # Reorder arch_results to match the visual flow (Method first, then all sizes)
            arch_results = []
            for method in selected_methods:
                for m in arch_models:
                    res = next((r for r in m["results"] if r["Method"].lower() == method.lower()), None)
                    if res: arch_results.append(res)
            
            if arch_results:
                raw_df = presentation_df(pd.DataFrame(arch_results))
                display_cols = [c for c in ["Method", "Resolution", ATTR_RUNTIME_COL, "Attribution Runtime Std (sec)", "Estimated Energy Consumption (kWh)", ATTR_MEMORY_COL, "Attribution Memory Std (MB)"] if c in raw_df.columns]
                
                qm_cols = ["Gini Index", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)", "Infidelity"]
                for qm_col in qm_cols:
                    if qm_col in raw_df.columns and raw_df[qm_col].notna().any():
                        display_cols.append(qm_col)
                            
                if "Status" in raw_df.columns and raw_df["Status"].astype(str).str.startswith("Failed").any():
                    display_cols.append("Status")
                st.table(style_dataframe(raw_df[display_cols], raw_precision=True))
