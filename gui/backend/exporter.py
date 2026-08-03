"""
Generates PDF and CSV reports for benchmark results.
Handles data formatting and visual report generation.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

ATTR_RUNTIME_COL = "Attribution Runtime (sec)"
ATTR_MEMORY_COL = "Peak Attribution Memory (MB)"
LEGACY_RUNTIME_COL = "Runtime (sec)"
LEGACY_MEMORY_COL = "Peak Memory (MB)"
METADATA_COLS = ["Timing Scope", "Memory Scope", "Model Cache"]
PRESENTATION_COL_ORDER = [
    "Method",
    "Model",
    "Input Size (px)",
    "Original Resolution",
    "Prediction",
    "Device",
    "Warmup Runs",
    "Memory Runs",
    "Measured Runs",
    ATTR_RUNTIME_COL,
    "Attribution Runtime Median (sec)",
    "Attribution Runtime Mean (sec)",
    "Attribution Runtime Std (sec)",
    "Attribution Runtime Min (sec)",
    "Attribution Runtime Max (sec)",
    "Estimated Energy Consumption (kWh)",
    ATTR_MEMORY_COL,
    "Gini Index",
    "Deletion AUC",
    "Insertion AUC",
    "Sensitivity (Max)",
    "Infidelity",
    "Status",
]

def normalize_metric_columns(df):
    df = df.copy()
    if ATTR_RUNTIME_COL not in df.columns and LEGACY_RUNTIME_COL in df.columns:
        df[ATTR_RUNTIME_COL] = df[LEGACY_RUNTIME_COL]
    if ATTR_MEMORY_COL not in df.columns and LEGACY_MEMORY_COL in df.columns:
        df[ATTR_MEMORY_COL] = df[LEGACY_MEMORY_COL]
    return df

def add_input_size_column(df):
    df = df.copy()
    if "Input Size (px)" not in df.columns and "Resolution" in df.columns:
        df["Input Size (px)"] = pd.to_numeric(df["Resolution"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return df

def presentation_df(df):
    df = add_input_size_column(normalize_metric_columns(df))
    duplicate_cols = [
        LEGACY_RUNTIME_COL, "Runtime Median (sec)", "Runtime Mean (sec)",
        "Runtime Std (sec)", "Runtime Min (sec)", "Runtime Max (sec)",
        LEGACY_MEMORY_COL, "Resolution"
    ] + METADATA_COLS
    df = df.drop(columns=[c for c in duplicate_cols if c in df.columns], errors="ignore")
    ordered_cols = [c for c in PRESENTATION_COL_ORDER if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + remaining_cols]

# Mapping to abbreviate long table headers in PDF tables so columns never clip
PDF_COLUMN_HEADER_MAP = {
    "Input Size (px)": "Size (px)",
    "Attribution Runtime (sec)": "Runtime (s)",
    "Attribution Runtime Median (sec)": "Runtime (s)",
    "Estimated Energy Consumption (kWh)": "Energy (kWh)",
    "Peak Attribution Memory (MB)": "Memory (MB)",
    "Warmup Runs": "Warmups",
    "Memory Runs": "Mem Runs",
    "Measured Runs": "Repeats",
    "Sensitivity (Max)": "Sensitivity",
}

def generate_pdf_report(batch_id, results_data, selected_methods, output_path, total_time=0, environment=None, benchmark_settings=None):
    """
    Generates a comprehensive PDF report:
    - Page 1: Environment & Benchmark Configuration Metadata + Overall Performance Evaluation
    - Pages 2+: Per-Image & Architecture Evaluation Pages (Heatmap Collage + Metrics Table)
    """
    with PdfPages(output_path) as pdf:
        # =========================================================================
        # --- PAGE 1: ENVIRONMENT, BENCHMARK CONFIGURATION & PERFORMANCE SUMMARY ---
        # =========================================================================
        fig_p1 = plt.figure(figsize=(11.69, 8.27)) # A4 Landscape
        
        # 1. Main Header
        fig_p1.text(0.5, 0.94, f"XAI Efficiency Benchmark — Report ({batch_id})", fontsize=18, fontweight='bold', ha='center', color='#0d9488')
        
        if total_time:
            h = int(total_time // 3600)
            m = int((total_time % 3600) // 60)
            s = int(total_time % 60)
            time_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s" if m > 0 else f"{total_time:.1f}s"
            fig_p1.text(0.5, 0.90, f"Total Batch Wall Time: {time_str}", fontsize=11, ha='center', fontweight='bold', color='#334155')

        # 2. Environment Summary (Left Column Top) & Benchmark Settings (Right Column Top)
        ax_env = fig_p1.add_axes([0.06, 0.52, 0.42, 0.33])
        ax_env.axis('off')
        ax_env.set_title("Environment Summary", fontsize=11, fontweight='bold', pad=8, color='#0f172a', loc='left')
        
        env = environment or {}
        gpu_list = []
        for d in env.get("cuda_devices", []):
            name = d.get("name", "Unknown GPU")
            tdp = d.get("tdp_w")
            gpu_list.append(f"{name} ({tdp}W TDP)" if tdp else name)
        gpu_names = ", ".join(gpu_list) or "N/A"
        cpu_display = env.get("processor", "Unknown CPU")
        if env.get("cpu_tdp_w"):
            cpu_display = f"{cpu_display} ({env.get('cpu_tdp_w')}W TDP)"

        env_table_data = [
            ["Platform / OS", str(env.get("platform", "Unknown"))],
            ["Python Version", str(env.get("python_version", "Unknown"))],
            ["PyTorch Version", str(env.get("torch_version", "Unknown"))],
            ["CUDA Version", str(env.get("torch_cuda_version") or "N/A")],
            ["Active Device", str(env.get("selected_device", "Unknown"))],
            ["CPU Model", cpu_display],
            ["GPU Model(s)", gpu_names],
        ]
        tbl_env = ax_env.table(cellText=env_table_data, colLabels=["Parameter", "Details"], loc='upper left', cellLoc='left')
        tbl_env.auto_set_font_size(False)
        tbl_env.set_fontsize(7.5)
        tbl_env.scale(1, 1.25)
        for (r, c), cell in tbl_env.get_celld().items():
            cell.set_edgecolor('#cbd5e1')
            if r == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#0d9488')
            else:
                if c == 0: cell.set_text_props(weight='bold')
                if r % 2 == 0: cell.set_facecolor('#f8fafc')

        # Benchmark Settings Table (Right Column Top)
        ax_cfg = fig_p1.add_axes([0.52, 0.52, 0.42, 0.33])
        ax_cfg.axis('off')
        ax_cfg.set_title("Benchmark Configuration", fontsize=11, fontweight='bold', pad=8, color='#0f172a', loc='left')
        
        cfg = benchmark_settings or {}
        # Collect models, methods, sizes from results if missing
        all_r_temp = []
        for g in results_data:
            for m in g["models"]: all_r_temp.extend(m["results"])
        fdf_temp = pd.DataFrame(all_r_temp) if all_r_temp else pd.DataFrame()
        
        models_list = sorted(list(fdf_temp["Model"].unique())) if "Model" in fdf_temp.columns else []
        methods_list = sorted(list(fdf_temp["Method"].unique())) if "Method" in fdf_temp.columns else []
        sizes_list = sorted(list(fdf_temp["Input Size (px)"].unique())) if "Input Size (px)" in fdf_temp.columns else []
        
        cfg_table_data = [
            ["Images Analyzed", str(len(results_data)), "-"],
            ["Models Selected", str(len(models_list)), ", ".join(models_list)],
            ["Methods Selected", str(len(methods_list)), ", ".join(methods_list)],
            ["Input Resolutions", str(len(sizes_list)), ", ".join([str(s) for s in sizes_list])],
            ["Warmup Runs", "-", str(cfg.get("warmup_runs", "-"))],
            ["Measured Repeats", "-", str(cfg.get("repeat_count", "-"))],
            ["Memory Runs", "-", str(cfg.get("memory_runs", "-"))],
            ["Task Order Strategy", "-", str(cfg.get("run_order", "Balanced"))],
        ]
        tbl_cfg = ax_cfg.table(cellText=cfg_table_data, colLabels=["Parameter", "Count", "Details"], loc='upper left', cellLoc='left')
        tbl_cfg.auto_set_font_size(False)
        tbl_cfg.set_fontsize(7.5)
        tbl_cfg.scale(1, 1.25)
        for (r, c), cell in tbl_cfg.get_celld().items():
            cell.set_edgecolor('#cbd5e1')
            if r == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#0d9488')
            else:
                if c == 0: cell.set_text_props(weight='bold')
                if r % 2 == 0: cell.set_facecolor('#f8fafc')

        # 3. Overall Performance Evaluation Summary Table (Bottom Half)
        if not fdf_temp.empty:
            ax_perf = fig_p1.add_axes([0.06, 0.06, 0.88, 0.38])
            ax_perf.axis('off')
            ax_perf.set_title("Overall Performance & Efficiency Evaluation Summary", fontsize=11, fontweight='bold', pad=8, color='#0f172a', loc='left')
            
            fdf_norm = normalize_metric_columns(fdf_temp)
            group_cols = ["Model"]
            if "Input Size (px)" in fdf_norm.columns:
                group_cols.append("Input Size (px)")
            elif "Resolution" in fdf_norm.columns:
                group_cols.append("Resolution")
                
            agg_dict = {ATTR_RUNTIME_COL: ["mean", "std"]}
            if ATTR_MEMORY_COL in fdf_norm.columns:
                agg_dict[ATTR_MEMORY_COL] = "mean"
            if "Estimated Energy Consumption (kWh)" in fdf_norm.columns:
                agg_dict["Estimated Energy Consumption (kWh)"] = "mean"

            perf_df = fdf_norm.groupby(group_cols).agg(agg_dict).reset_index()
            # Flatten multi-index columns
            flat_cols = []
            for col in perf_df.columns:
                if isinstance(col, tuple):
                    if col[1]:
                        flat_cols.append(f"{col[0]} ({col[1]})")
                    else:
                        flat_cols.append(col[0])
                else:
                    flat_cols.append(col)
            perf_df.columns = flat_cols

            # Abbreviate and round columns
            for c in perf_df.columns:
                if "Runtime" in c or "Energy" in c:
                    perf_df[c] = perf_df[c].apply(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) and pd.notna(v) else str(v))
                elif "Memory" in c:
                    perf_df[c] = perf_df[c].apply(lambda v: f"{v:.1f}" if isinstance(v, (int, float)) and pd.notna(v) else str(v))

            abbrev_cols = [PDF_COLUMN_HEADER_MAP.get(c, c) for c in perf_df.columns]
            tbl_perf = ax_perf.table(cellText=perf_df.values, colLabels=abbrev_cols, loc='upper center', cellLoc='center')
            tbl_perf.auto_set_font_size(False)
            tbl_perf.set_fontsize(7.5)
            tbl_perf.scale(1, 1.35)
            for (r, c), cell in tbl_perf.get_celld().items():
                cell.set_edgecolor('#cbd5e1')
                if r == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#0d9488')
                else:
                    if r % 2 == 0: cell.set_facecolor('#f8fafc')

        pdf.savefig(fig_p1)
        plt.close()

        # =========================================================================
        # --- PAGES 2+: PER-IMAGE & ARCHITECTURE EVALUATIONS ---
        # =========================================================================
        for group in results_data:
            img_idx = group['img_idx']
            architectures = []
            for m in group["models"]:
                if m["model"] not in architectures:
                    architectures.append(m["model"])
            
            for arch in architectures:
                fig = plt.figure(figsize=(11.69, 8.27)) # A4 Landscape
                
                # GridSpec layout with ample margins to prevent clipping top/right
                gs = gridspec.GridSpec(
                    3, 2,
                    height_ratios=[0.06, 0.54, 0.40],
                    width_ratios=[1, 3.2],
                    left=0.06, right=0.94, bottom=0.05, top=0.93,
                    wspace=0.15, hspace=0.25
                )
                
                # Header
                fig.text(0.06, 0.95, f"Image {img_idx}  |  Architecture: {arch}", fontsize=13, fontweight='bold', ha='left', color='#0f172a')
                
                arch_models = [m for m in group["models"] if m["model"] == arch]
                sample_m = arch_models[0]
                
                # 1. INPUT IMAGE (Left Column)
                ax_input = fig.add_subplot(gs[1, 0])
                img_path = os.path.join(sample_m["session_dir"], "input_image.jpg")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    ax_input.imshow(img)
                    orig_res = sample_m["results"][0].get("Original Resolution", "N/A")
                    ax_input.set_title(f"Input Image ({orig_res})", fontsize=8.5, pad=4, color='#334155', fontweight='bold')
                ax_input.axis('off')

                # 2. HEATMAP COLLAGE (Right Column)
                num_methods = len(selected_methods)
                num_sizes = len(arch_models)
                gs_inner = gridspec.GridSpecFromSubplotSpec(
                    num_methods, num_sizes,
                    subplot_spec=gs[1, 1],
                    hspace=0.25, wspace=0.08
                )
                
                for m_idx, method in enumerate(selected_methods):
                    for s_idx, m_data in enumerate(arch_models):
                        ax_hm = fig.add_subplot(gs_inner[m_idx, s_idx])
                        res = next((r for r in m_data["results"] if r["Method"].lower() == method.lower()), None)
                        
                        if res:
                            h_p = os.path.join(m_data["session_dir"], "heatmaps", f"{res['Method']}.png")
                            if os.path.exists(h_p):
                                hm_img = Image.open(h_p)
                                ax_hm.imshow(hm_img)
                                label = f"{m_data['input_size']}px"
                                if s_idx == 0:
                                    label = f"{method}\n{label}"
                                ax_hm.set_title(label, fontsize=7.5, pad=2, color='#1e293b')
                        ax_hm.axis('off')

                # 3. CONSOLIDATED METRICS TABLE (Bottom)
                ax_table = fig.add_subplot(gs[2, :])
                ax_table.axis('off')
                
                arch_results = []
                for method in selected_methods:
                    for m in arch_models:
                        res = next((r for r in m["results"] if r["Method"].lower() == method.lower()), None)
                        if res:
                            arch_results.append(res)
                
                if arch_results:
                    df = presentation_df(pd.DataFrame(arch_results))
                    raw_cols = ["Method", "Input Size (px)", "Prediction", "Warmup Runs", "Memory Runs", "Measured Runs", ATTR_RUNTIME_COL, "Estimated Energy Consumption (kWh)", ATTR_MEMORY_COL]
                    if "Gini Index" in df.columns and any(df["Gini Index"].notna()): raw_cols.append("Gini Index")
                    if "Deletion AUC" in df.columns and any(df["Deletion AUC"].notna()): raw_cols.append("Deletion AUC")
                    if "Insertion AUC" in df.columns and any(df["Insertion AUC"].notna()): raw_cols.append("Insertion AUC")
                    if "Sensitivity (Max)" in df.columns and any(df["Sensitivity (Max)"].notna()): raw_cols.append("Sensitivity (Max)")
                    if "Infidelity" in df.columns and any(df["Infidelity"].notna()): raw_cols.append("Infidelity")
                    raw_cols = [c for c in raw_cols if c in df.columns]
                    if "Status" in df.columns and any(df["Status"].notna()): raw_cols.append("Status")
                    
                    display_df = df[raw_cols].copy()
                    for c in display_df.columns:
                        if "Runtime" in c or "Energy" in c or "AUC" in c or "Gini" in c or "Sensitivity" in c or "Infidelity" in c:
                            display_df[c] = display_df[c].apply(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) and pd.notna(v) else ("-" if pd.isna(v) else str(v)))
                        elif "Memory" in c:
                            display_df[c] = display_df[c].apply(lambda v: f"{v:.1f}" if isinstance(v, (int, float)) and pd.notna(v) else ("-" if pd.isna(v) else str(v)))

                    # Map header titles to compact names so they never overflow table columns
                    abbrev_headers = [PDF_COLUMN_HEADER_MAP.get(c, c) for c in raw_cols]
                    table_data = display_df.fillna("-").values
                    
                    tbl = ax_table.table(cellText=table_data, colLabels=abbrev_headers, loc='upper center', cellLoc='center')
                    tbl.auto_set_font_size(False)
                    
                    font_size = 7.5 if len(raw_cols) <= 8 else (6.5 if len(raw_cols) <= 11 else 5.5)
                    tbl.set_fontsize(font_size)
                    tbl.scale(1, 1.35)
                    
                    for (row, col_i), cell in tbl.get_celld().items():
                        cell.set_edgecolor('#cbd5e1')
                        if row == 0:
                            cell.set_text_props(weight='bold', color='white')
                            cell.set_facecolor('#0d9488')
                        else:
                            if row % 2 == 0:
                                cell.set_facecolor('#f8fafc')

                pdf.savefig(fig)
                plt.close()

def generate_csv_report(results_data, output_path):
    all_r = []
    for g in results_data:
        for m in g["models"]: all_r.extend(m["results"])
    if all_r:
        presentation_df(pd.DataFrame(all_r)).to_csv(output_path, index=False)
        return True
    return False
