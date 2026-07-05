import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np

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
    ATTR_MEMORY_COL,
    "Gini Index",
    "Deletion AUC",
    "Insertion AUC",
    "Infidelity",
    "Quality Eval Time (sec)",
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

def generate_pdf_report(batch_id, results_data, selected_methods, output_path, total_time=0, environment=None):
    """
    Generates a PDF report that exactly mirrors the GUI's compact collage style.
    """
    with PdfPages(output_path) as pdf:
        # ... (Loop over content pages remains same)
        for group in results_data:
            img_idx = group['img_idx']
            architectures = []
            for m in group["models"]:
                if m["model"] not in architectures: architectures.append(m["model"])
            
            for arch in architectures:
                # Create a wide A4-style landscape page
                fig = plt.figure(figsize=(11.69, 8.27)) # A4 Landscape
                
                # Use GridSpec to define sections:
                # Top: Architecture Name
                # Middle: Input Image (Left) and Heatmap Collage (Right)
                # Bottom: Metrics Table
                gs = gridspec.GridSpec(3, 2, height_ratios=[0.05, 0.6, 0.35], width_ratios=[1, 3], 
                                     left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.1, hspace=0.2)
                
                # --- 0. HEADER ---
                plt.figtext(0.05, 0.96, f"Image {img_idx} | Architecture: {arch}", fontsize=14, fontweight='bold', ha='left')
                
                arch_models = [m for m in group["models"] if m["model"] == arch]
                sample_m = arch_models[0]
                
                # --- 1. INPUT IMAGE (Left Column) ---
                ax_input = fig.add_subplot(gs[1, 0])
                img_path = os.path.join(sample_m["session_dir"], "input_image.jpg")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    ax_input.imshow(img)
                    orig_res = sample_m["results"][0].get("Original Resolution", "N/A")
                    ax_input.set_title(f"Input ({orig_res})", fontsize=10, pad=5)
                ax_input.axis('off')

                # --- 2. HEATMAP COLLAGE (Right Column) ---
                # Sub-grid for the methods and sizes
                num_methods = len(selected_methods)
                num_sizes = len(arch_models)
                gs_inner = gridspec.GridSpecFromSubplotSpec(num_methods, num_sizes, subplot_spec=gs[1, 1], hspace=0.3, wspace=0.05)
                
                for m_idx, method in enumerate(selected_methods):
                    for s_idx, m_data in enumerate(arch_models):
                        ax_hm = fig.add_subplot(gs_inner[m_idx, s_idx])
                        res = next((r for r in m_data["results"] if r["Method"].lower() == method.lower()), None)
                        
                        if res:
                            h_p = os.path.join(m_data["session_dir"], "heatmaps", f"{res['Method']}.png")
                            if os.path.exists(h_p):
                                hm_img = Image.open(h_p)
                                ax_hm.imshow(hm_img)
                                # Show method name only on the first size of each row
                                label = f"{m_data['input_size']}px"
                                if s_idx == 0: label = f"{method}\n{label}"
                                ax_hm.set_title(label, fontsize=8, pad=2)
                        ax_hm.axis('off')

                # --- 3. CONSOLIDATED TABLE (Bottom) ---
                ax_table = fig.add_subplot(gs[2, :])
                ax_table.axis('off')
                
                arch_results = []
                # Follow Method -> Size order exactly like GUI
                for method in selected_methods:
                    for m in arch_models:
                        res = next((r for r in m["results"] if r["Method"].lower() == method.lower()), None)
                        if res: arch_results.append(res)
                
                if arch_results:
                    df = presentation_df(pd.DataFrame(arch_results))
                    cols = ["Method", "Input Size (px)", "Prediction", "Warmup Runs", "Memory Runs", "Measured Runs", ATTR_RUNTIME_COL, ATTR_MEMORY_COL]
                    if "Gini Index" in df.columns and any(df["Gini Index"].notna()): cols.append("Gini Index")
                    if "Deletion AUC" in df.columns and any(df["Deletion AUC"].notna()): cols.append("Deletion AUC")
                    if "Insertion AUC" in df.columns and any(df["Insertion AUC"].notna()): cols.append("Insertion AUC")
                    if "Infidelity" in df.columns and any(df["Infidelity"].notna()): cols.append("Infidelity")
                    if "Quality Eval Time (sec)" in df.columns and any(df["Quality Eval Time (sec)"].notna()): cols.append("Quality Eval Time (sec)")
                    # Only keep columns that exist in data
                    cols = [c for c in cols if c in df.columns]
                    if "Status" in df.columns and any(df["Status"].notna()): cols.append("Status")
                    
                    table_data = df[cols].fillna("-").values
                    tbl = ax_table.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center')
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(9)
                    tbl.scale(1, 1.4)
                    
                    # Style: Blue Header, white text
                    for (row, col), cell in tbl.get_celld().items():
                        if row == 0:
                            cell.set_text_props(weight='bold', color='white')
                            cell.set_facecolor('#2E86C1')

                pdf.savefig(fig)
                plt.close()

        # --- SUMMARY PAGE ---
        all_r = []
        for g in results_data:
            for m in g["models"]: all_r.extend(m["results"])
        
        if all_r:
            fdf = pd.DataFrame(all_r)
            fig_sum = plt.figure(figsize=(11.69, 8.27))
            plt.text(0.5, 0.92, "Aggregate Efficiency Summary", fontsize=20, fontweight='bold', ha='center', color='#2E86C1')
            
            if total_time:
                h = int(total_time // 3600)
                m = int((total_time % 3600) // 60)
                s = int(total_time % 60)
                time_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s" if m > 0 else f"{total_time:.1f}s"
                plt.text(0.5, 0.87, f"Batch Wall Time: {time_str}", fontsize=12, ha='center', fontweight='bold')

            if environment:
                gpu_names = ", ".join([d.get("name", "Unknown GPU") for d in environment.get("cuda_devices", [])]) or "None"
                env_lines = [
                    f"Git: {environment.get('git_commit', 'unknown')}",
                    f"Python: {environment.get('python_version', 'unknown')}",
                    f"Torch: {environment.get('torch_version', 'unknown')}",
                    f"CUDA: {environment.get('torch_cuda_version') or 'not available'}",
                    f"Device: {environment.get('selected_device', 'unknown')}",
                    f"GPU(s): {gpu_names}",
                ]
                plt.text(0.5, 0.80, " | ".join(env_lines), fontsize=8, ha='center', wrap=True)

            ax_sum_tbl = fig_sum.add_axes([0.1, 0.1, 0.8, 0.7])
            ax_sum_tbl.axis('off')
            group_cols = ["Model", "Resolution"]
            if "Original Resolution" in fdf.columns: group_cols.append("Original Resolution")
            
            fdf = normalize_metric_columns(fdf)
            summary_df = fdf.groupby(group_cols).agg({ATTR_RUNTIME_COL: "mean", ATTR_MEMORY_COL: "mean"}).reset_index()
            tbl_sum = ax_sum_tbl.table(cellText=summary_df.values, colLabels=summary_df.columns, loc='center', cellLoc='center')
            tbl_sum.auto_set_font_size(False)
            tbl_sum.set_fontsize(9)
            tbl_sum.scale(1, 1.8)
            
            for (row, col), cell in tbl_sum.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#2E86C1')

            pdf.savefig(fig_sum)
            plt.close()

def generate_csv_report(results_data, output_path):
    all_r = []
    for g in results_data:
        for m in g["models"]: all_r.extend(m["results"])
    if all_r:
        presentation_df(pd.DataFrame(all_r)).to_csv(output_path, index=False)
        return True
    return False
