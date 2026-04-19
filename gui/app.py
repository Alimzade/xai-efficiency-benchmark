import streamlit as st
import os
import pandas as pd
import torch
import platform
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from PIL import Image
import requests
from io import BytesIO
from session_manager import SessionManager
from benchmark_runner import run_benchmark_task
from exporter import generate_pdf_report, generate_csv_report

# --- SILENCE NOISY WARNINGS ---
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum.attr._utils.visualization")

# --- Page Config ---
st.set_page_config(page_title="XAI Efficiency Benchmark", page_icon="🔍", layout="wide")

# CSS
st.markdown("""
    <style>
    .compact-preview { max-width: 300px; margin-left: auto; margin-right: 0; }
    div[data-testid="column"]:nth-child(3) { display: flex; flex-direction: column; align-items: flex-end; }
    .stButton button { padding: 2px 10px !important; font-size: 0.9em !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { border: none !important; }
    .status-pulse { color: #ff4b4b; font-weight: bold; animation: pulse 1.5s infinite; font-size: 1.1em; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
    </style>
    """, unsafe_allow_html=True)

sm = SessionManager()

# --- HELPER FUNCTIONS ---

def get_cpu_info(): return platform.processor() or "Generic CPU"
def get_base64(img):
    buffered = BytesIO(); img.save(buffered, format="PNG")
    import base64; return base64.b64encode(buffered.getvalue()).decode()

def style_dataframe(df):
    subset_cols = [c for c in ["Runtime (sec)", "Peak Memory (MB)", "Avg Runtime (sec)", "Avg Peak Memory (MB)"] if c in df.columns]
    return df.style.background_gradient(cmap="coolwarm", subset=subset_cols).format({c: "{:.4f}" if "sec" in c else "{:.2f}" for c in subset_cols})

def plot_method_runtime_log(df, title="Runtime Comparison (Log Scale)"):
    fig, ax = plt.subplots(figsize=(10, 6))
    summary = df.groupby("Method")["Runtime (sec)"].mean().sort_values().reset_index()
    sns.barplot(data=summary, x="Runtime (sec)", y="Method", palette="crest", ax=ax, edgecolor="black")
    ax.set_xscale("log"); ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    ax.grid(True, ls="-", alpha=0.2); plt.tight_layout(); return fig

def plot_model_comparison_grouped(df, title="Architecture Efficiency Comparison"):
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df, x="Method", y="Runtime (sec)", hue="Model", palette="colorblind", ax=ax, edgecolor="black")
    ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    plt.xticks(rotation=45); ax.legend(loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout(); return fig

def render_result_group(group, selected_methods):
    with st.expander(f"🖼️ Results for Image {group['img_idx']}", expanded=True):
        # Group entries by base model architecture
        architectures = []
        for m in group["models"]:
            if m["model"] not in architectures: architectures.append(m["model"])
        
        for arch in architectures:
            st.markdown(f"#### Model: `{arch}`")
            arch_models = [m for m in group["models"] if m["model"] == arch]
            
            # Layout: Input Image (Left) | Method Collage (Right)
            col_left, col_right = st.columns([1, 3])
            
            # 1. Show Input Image once for this Architecture
            sample_m = arch_models[0]
            img_path = os.path.join(sample_m["session_dir"], "input_image.jpg")
            if os.path.exists(img_path):
                # Pull original resolution from the first result entry
                orig_res = sample_m["results"][0].get("Original Resolution", "Unknown") if sample_m["results"] else "Unknown"
                col_left.image(img_path, caption=f"Input Image ({orig_res})", use_container_width=True)
            
            # 2. Show Heatmap Rows (One row per Method, sizes side-by-side)
            with col_right:
                for method in selected_methods:
                    st.markdown(f"**{method}**")
                    # Create at least 4 columns to ensure thumbnails stay small and separate
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
                                st.markdown("<div style='height: 60px; border: 1px dashed gray; text-align: center; padding-top: 20px; color: gray; font-size: 0.7em;'>...</div>", unsafe_allow_html=True)
            
            # 3. Consolidated Table for all sizes of this Architecture
            # Reorder arch_results to match the visual flow (Method first, then all sizes)
            arch_results = []
            for method in selected_methods:
                for m in arch_models:
                    res = next((r for r in m["results"] if r["Method"].lower() == method.lower()), None)
                    if res: arch_results.append(res)
            
            if arch_results:
                st.table(style_dataframe(pd.DataFrame(arch_results)))

@st.dialog("Image Viewer", width="large")
def show_lightbox(img):
    st.markdown(f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{st.session_state.current_img_base64}" style="max-height: 80vh; max-width: 100%; object-fit: contain;"></div>', unsafe_allow_html=True)

# --- Initialize Session State ---
if 'img_idx' not in st.session_state: st.session_state.img_idx = 0
if 'persisted_urls' not in st.session_state: st.session_state.persisted_urls = ""
if 'last_run_results' not in st.session_state: st.session_state.last_run_results = []
if 'stop_requested' not in st.session_state: st.session_state.stop_requested = False
if 'is_finished' not in st.session_state: st.session_state.is_finished = False
if 'balloons_triggered' not in st.session_state: st.session_state.balloons_triggered = False
if 'benchmark_running' not in st.session_state: st.session_state.benchmark_running = False
if 'run_progress_idx' not in st.session_state: st.session_state.run_progress_idx = 0
if 'current_batch_id' not in st.session_state: st.session_state.current_batch_id = ""
if 'current_img_base64' not in st.session_state: st.session_state.current_img_base64 = ""

# --- SIDEBAR ---
st.sidebar.title("Benchmark Settings ⚙️")
is_running = st.session_state.benchmark_running

model_opts = ['resnet50', 'convnext-t', 'efficientnet-b0', 'swin-t', 'regnet-y-8gf', 'mobilenet-v3-large', 'densenet121', 'vit-b-16']
selected_models = st.sidebar.multiselect("Model Architectures", model_opts, default=["resnet50"], disabled=is_running)

# Input Size Selection
fixed_size_trigger = any(m in ["vit-b-16", "swin-t"] for m in selected_models)
if fixed_size_trigger:
    selected_sizes = [224]
    st.sidebar.text_input("Input Sizes (px)", value="224", disabled=True)
    st.sidebar.caption("⚠️ *Fixed-size architecture selected (Locked to 224px)*")
else:
    size_str = st.sidebar.text_input("Input Sizes (px)", value="224", help="Only applicable to CNN-based architectures.", disabled=is_running)
    st.sidebar.markdown('<div style="margin-top: -15px; margin-bottom: 15px; font-size: 0.85em; color: gray;">Separate by commas (e.g., 224, 448, 512).</div>', unsafe_allow_html=True)
    try:
        selected_sizes = [int(s.strip()) for s in size_str.split(",") if s.strip().isdigit()]
        if not selected_sizes: selected_sizes = [224]
    except:
        selected_sizes = [224]
        st.sidebar.error("Invalid size format. Using 224.")

xai_opts = ["Saliency", "Integrated_Gradients", "Guided_Backprop", "Input_X_Gradient"]
selected_methods = st.sidebar.multiselect("XAI Methods", xai_opts, default=["Saliency", "Integrated_Gradients"], disabled=is_running)
st.sidebar.divider(); st.sidebar.subheader("System Status")
selected_device_mode = st.sidebar.radio("Force execution on:", ["GPU (CUDA)" if torch.cuda.is_available() else "CPU", "CPU"] if torch.cuda.is_available() else ["CPU"], label_visibility="collapsed", disabled=is_running)
if "GPU" in selected_device_mode: st.sidebar.success(f"**GPU:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Active'}")
else: st.sidebar.warning(f"**CPU:** {get_cpu_info()}")
st.sidebar.divider(); st.sidebar.info("### How to Cite")
st.sidebar.code("""@software{alimzade2025xai,
  author  = {Anar Alimzade},
  title   = {Efficiency Benchmark for XAI},
  year    = {2025},
  month   = {May},
  version = {1.0.0}
}""", language="bibtex")

# --- Tabs ---
tab1, tab2 = st.tabs(["🚀 Run Benchmark", "📜 Results History"])

with tab1:
    col_input, col_spacer, col_preview = st.columns([2.5, 0.2, 0.8])
    with col_input:
        uploaded_files = st.file_uploader("Drag and drop images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, disabled=is_running)
        with st.expander("Paste Image URLs", expanded=False):
            urls_input = st.text_area("Input URLs here", st.session_state.persisted_urls, height=100, label_visibility="collapsed", disabled=is_running)
            st.session_state.persisted_urls = urls_input
        url_list = [u.strip() for u in st.session_state.persisted_urls.split("\n") if u.strip()]
        img_sources = (uploaded_files if uploaded_files else []) + url_list

    with col_preview:
        if img_sources:
            num_imgs = len(img_sources)
            if st.session_state.img_idx >= num_imgs: st.session_state.img_idx = 0
            st.markdown('<div class="compact-preview">', unsafe_allow_html=True)
            current_src = img_sources[st.session_state.img_idx]
            try:
                if hasattr(current_src, 'name'): img_view = Image.open(current_src)
                else: response = requests.get(current_src); img_view = Image.open(BytesIO(response.content))
                st.session_state.current_img_base64 = get_base64(img_view)
                st.markdown(f"<div style='text-align: center; color: gray; font-size: 0.8em; margin-bottom: 2px;'>Resolution: {img_view.size[0]}x{img_view.size[1]} px</div>", unsafe_allow_html=True)
                if st.button("View", width='stretch'): show_lightbox(img_view)
                with st.container(height=310, border=False): st.image(img_view, width='stretch')
            except: st.markdown("<div style='height: 330px; text-align: center; padding-top: 100px;'>Preview unavailable</div>", unsafe_allow_html=True)
            n1, n2, n3 = st.columns([1, 0.8, 1])
            with n1:
                if st.button("⬅️ Prev", key="prev_btn", width='stretch'): st.session_state.img_idx = (st.session_state.img_idx - 1) % num_imgs; st.rerun()
            with n2: st.markdown(f"<div style='text-align: center; padding-top: 5px; font-weight: bold;'>{st.session_state.img_idx + 1}/{num_imgs}</div>", unsafe_allow_html=True)
            with n3:
                if st.button("Next ➡️", key="next_btn", width='stretch'): st.session_state.img_idx = (st.session_state.img_idx + 1) % num_imgs; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    
    # --- ACTION BUTTONS ---
    if not st.session_state.benchmark_running:
        if st.button("Start Multi-Model Benchmark ⚡", width='stretch'):
            st.session_state.last_run_results = []
            st.session_state.is_finished = False
            if not img_sources or not selected_models or not selected_methods: st.error("Select Settings.")
            else:
                st.session_state.current_batch_id = sm.start_batch()
                st.session_state.stop_requested = False; st.session_state.balloons_triggered = False
                st.session_state.run_progress_idx = 0; st.session_state.benchmark_running = True; st.rerun()
    else:
        if st.button("🛑 Stop Benchmark", width='stretch'):
            st.session_state.stop_requested = True; st.session_state.benchmark_running = False
            st.session_state.last_run_results = []; st.rerun()

    # --- RESULTS AREA ---
    if st.session_state.benchmark_running and not st.session_state.is_finished:
        total_steps = len(img_sources) * len(selected_models) * len(selected_sizes) * len(selected_methods)
        idx = st.session_state.run_progress_idx
        img_i = idx // (len(selected_models) * len(selected_methods) * len(selected_sizes))
        rem = idx % (len(selected_models) * len(selected_methods) * len(selected_sizes))

        mod_i = rem // (len(selected_methods) * len(selected_sizes))
        rem = rem % (len(selected_methods) * len(selected_sizes))

        met_i = rem // len(selected_sizes)
        size_i = rem % len(selected_sizes)

        cur_mod = selected_models[mod_i] if mod_i < len(selected_models) else "?"
        cur_met = selected_methods[met_i] if met_i < len(selected_methods) else "?"
        cur_size = selected_sizes[size_i] if size_i < len(selected_sizes) else "?"

        st.markdown(f"<div class='status-pulse'>🚀 STEP {idx + 1}/{total_steps}: Running {cur_met} on {cur_mod} @ {cur_size}px (Image {img_i + 1})</div>", unsafe_allow_html=True)
        st.progress(idx / total_steps); st.divider()


    if st.session_state.last_run_results:
        for group in st.session_state.last_run_results:
            render_result_group(group, selected_methods)

        if st.session_state.is_finished:
            all_r = []
            for g in st.session_state.last_run_results:
                for m in g["models"]: all_r.extend(m["results"])
            if all_r:
                fdf = pd.DataFrame(all_r)
                fdf["Model_Size"] = fdf["Model"] + " (" + fdf["Resolution"] + ")"
                
                st.divider()
                st.header("🔬 Batch Summary")
                
                # --- EXPORT BUTTONS ---
                ex1, ex2, ex3 = st.columns([1, 1, 3])
                with ex1:
                    csv_path = os.path.join(sm.base_dir, st.session_state.current_batch_id, f"{st.session_state.current_batch_id}.csv")
                    if generate_csv_report(st.session_state.last_run_results, csv_path):
                        with open(csv_path, "rb") as f:
                            st.download_button("📥 Export CSV", data=f, file_name=f"{st.session_state.current_batch_id}.csv", mime="text/csv", use_container_width=True)
                with ex2:
                    pdf_path = os.path.join(sm.base_dir, st.session_state.current_batch_id, f"{st.session_state.current_batch_id}.pdf")
                    # Use a spinner while generating PDF
                    with st.spinner("Generating PDF..."):
                        generate_pdf_report(st.session_state.current_batch_id, st.session_state.last_run_results, selected_methods, pdf_path)
                    with open(pdf_path, "rb") as f:
                        st.download_button("📄 Export PDF", data=f, file_name=f"{st.session_state.current_batch_id}.pdf", mime="application/pdf", use_container_width=True)

                cs1, cs2 = st.columns(2)
                with cs1:
                    st.subheader("Configuration Averages")
                    group_cols = ["Model", "Resolution"]
                    if "Original Resolution" in fdf.columns: group_cols.append("Original Resolution")
                    summary_df = fdf.groupby(group_cols).agg({"Runtime (sec)": "mean", "Peak Memory (MB)": "mean"}).reset_index()
                    st.table(style_dataframe(summary_df))
                    fig1, ax1 = plt.subplots(figsize=(12, 7))
                    sns.barplot(data=fdf, x="Method", y="Runtime (sec)", hue="Model_Size", palette="colorblind", ax=ax1, edgecolor="black")
                    ax1.set_title("Architecture & Resolution Efficiency", fontsize=14, fontweight='bold')
                    plt.xticks(rotation=45); ax1.legend(loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout()
                    st.pyplot(fig1)
                with cs2:
                    st.subheader("Method Averages"); st.table(style_dataframe(fdf.groupby("Method").agg({"Runtime (sec)": "mean", "Peak Memory (MB)": "mean"}).reset_index()))
                    st.pyplot(plot_method_runtime_log(fdf))
                if not st.session_state.balloons_triggered: st.balloons(); st.session_state.balloons_triggered = True

    # --- ENGINE ---
    if st.session_state.benchmark_running and not st.session_state.is_finished:
        steps_per_img = len(selected_models) * len(selected_methods) * len(selected_sizes)
        idx = st.session_state.run_progress_idx
        img_i = idx // steps_per_img
        rem = idx % steps_per_img

        mod_i = rem // (len(selected_methods) * len(selected_sizes))
        rem = rem % (len(selected_methods) * len(selected_sizes))

        met_i = rem // len(selected_sizes)
        size_i = rem % len(selected_sizes)

        if img_i < len(img_sources):

            if len(st.session_state.last_run_results) <= img_i:
                st.session_state.last_run_results.append({"img_idx": img_i + 1, "models": [], "source": img_sources[img_i]})
            
            src = img_sources[img_i]
            model_name = selected_models[mod_i]
            target_size = selected_sizes[size_i]
            method_name = selected_methods[met_i]
            
            target_group = st.session_state.last_run_results[img_i]
            model_label = f"{model_name} ({target_size}px)"
            model_entry = next((m for m in target_group["models"] if m.get("model_label") == model_label), None)
            
            if not model_entry:
                s_dir = sm.get_task_path(st.session_state.current_batch_id, img_i + 1, f"{model_name}_{target_size}")
                if hasattr(src, 'getbuffer'):
                    tp = os.path.join(s_dir, "input_image.jpg"); f = open(tp, "wb"); f.write(src.getbuffer()); f.close(); fs = tp
                else: fs = src
                model_entry = {"model": model_name, "model_label": model_label, "input_size": target_size, "results": [], "session_dir": s_dir, "src_path": fs}
                target_group["models"].append(model_entry)
            
            results = run_benchmark_task({
                "model_name": model_name, 
                "image_source": model_entry["src_path"], 
                "methods": [method_name.lower()], 
                "force_device": "cuda" if "GPU" in selected_device_mode else "cpu", 
                "input_size": target_size
            }, model_entry["session_dir"])
            
            model_entry["results"].extend(results)
            st.session_state.run_progress_idx += 1
            
            if st.session_state.run_progress_idx >= (len(img_sources) * steps_per_img):
                st.session_state.is_finished = True; st.session_state.benchmark_running = False
                clean_results = []
                for g in st.session_state.last_run_results:
                    cg = g.copy()
                    if hasattr(cg["source"], 'name'): cg["source"] = cg["source"].name
                    clean_results.append(cg)
                with open(os.path.join(sm.base_dir, st.session_state.current_batch_id, "batch_results.json"), 'w') as f:
                    json.dump({"results": clean_results, "methods": selected_methods}, f, indent=4)
            st.rerun()

with tab2:
    batches = sm.list_batches()
    if batches:
        bid = st.selectbox("Select Benchmark Batch", [b["id"] for b in batches])
        batch_meta_p = os.path.join(sm.base_dir, bid, "batch_results.json")
        if os.path.exists(batch_meta_p):
            try:
                with open(batch_meta_p, 'r') as f: meta = json.load(f)
                for group in meta["results"]: render_result_group(group, meta["methods"])
                all_h_r = []
                for g in meta["results"]:
                    for m in g["models"]: all_h_r.extend(m["results"])
                if all_h_r:
                    hdf = pd.DataFrame(all_h_r)
                    hdf["Model_Size"] = hdf["Model"] + " (" + hdf.get("Resolution", "224x224") + ")"
                    # Ensure consistent row ordering: Method -> Resolution
                    hdf = hdf.sort_values(by=["Method", "Resolution"])
                    
                    st.divider()
                    st.header("🔬 Batch Summary (Historical)")

                    # --- EXPORT BUTTONS (History) ---
                    hx1, hx2, hx3 = st.columns([1, 1, 3])
                    with hx1:
                        h_csv = os.path.join(sm.base_dir, bid, f"{bid}.csv")
                        if generate_csv_report(meta["results"], h_csv):
                            with open(h_csv, "rb") as f:
                                st.download_button("📥 Export CSV", data=f, file_name=f"{bid}.csv", mime="text/csv", key=f"csv_{bid}", use_container_width=True)
                    with hx2:
                        h_pdf = os.path.join(sm.base_dir, bid, f"{bid}.pdf")
                        with st.spinner("Generating PDF..."):
                            generate_pdf_report(bid, meta["results"], meta["methods"], h_pdf)
                        with open(h_pdf, "rb") as f:
                            st.download_button("📄 Export PDF", data=f, file_name=f"{bid}.pdf", mime="application/pdf", key=f"pdf_{bid}", use_container_width=True)

                    hc1, hc2 = st.columns(2)
                    with hc1:
                        st.subheader("Configuration Averages")
                        # Include Original Resolution in group by if it exists
                        group_cols = ["Model", "Resolution"]
                        if "Original Resolution" in hdf.columns: group_cols.append("Original Resolution")
                        h_summ = hdf.groupby(group_cols).agg({"Runtime (sec)": "mean", "Peak Memory (MB)": "mean"}).reset_index()
                        st.table(style_dataframe(h_summ))
                        fig_h, ax_h = plt.subplots(figsize=(12, 7))
                        sns.barplot(data=hdf, x="Method", y="Runtime (sec)", hue="Model_Size", palette="colorblind", ax=ax_h, edgecolor="black")
                        plt.xticks(rotation=45); ax_h.legend(loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout()
                        st.pyplot(fig_h)
                    with hc2:
                        st.subheader("Method Averages"); st.table(style_dataframe(hdf.groupby("Method").agg({"Runtime (sec)": "mean", "Peak Memory (MB)": "mean"}).reset_index()))
                        st.pyplot(plot_method_runtime_log(hdf))
                if st.button("🗑️ Delete Entire Batch"): sm.delete_batch(bid); st.rerun()
            except Exception as e:
                st.error(f"Error reading historical data: {str(e)}")
                if st.button("🗑️ Delete Corrupted Batch"): sm.delete_batch(bid); st.rerun()
        else: st.info("Loading metadata for this batch...")
    else:
        st.info("No benchmark history found. Start a new run in the 'Run Benchmark' tab!")
