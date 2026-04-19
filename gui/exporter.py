import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import numpy as np

def generate_pdf_report(batch_id, results_data, selected_methods, output_path):
    """
    Generates a PDF report that exactly mirrors the GUI's compact collage style.
    """
    with PdfPages(output_path) as pdf:
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
                    df = pd.DataFrame(arch_results)
                    cols = ["Method", "Resolution", "Prediction", "Runtime (sec)", "Peak Memory (MB)"]
                    # Only keep columns that exist in data
                    cols = [c for c in cols if c in df.columns]
                    if "Status" in df.columns and any(df["Status"].notna()): cols.append("Status")
                    
                    table_data = df[cols].values
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

def generate_csv_report(results_data, output_path):
    all_r = []
    for g in results_data:
        for m in g["models"]: all_r.extend(m["results"])
    if all_r:
        pd.DataFrame(all_r).to_csv(output_path, index=False)
        return True
    return False
