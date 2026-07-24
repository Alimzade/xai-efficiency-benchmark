"""
History Page Module
Renders the historical benchmarking results page, allowing users to compare past runs side-by-side.
"""
import os
import json
import pandas as pd
import streamlit as st

from gui.core import sm
from gui.utils.state import check_and_reset_session_after_delete
from gui.utils.helpers import format_time, format_run_timestamps, format_run_duration, get_batch_display_name, rerun_app
from gui.utils.processing import ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL, ATTR_MEMORY_COL, LEGACY_MEMORY_COL, metric_col, normalize_metric_columns
from gui.components.cards import render_analytics_sections, render_environment_summary, render_parameters_mapping_table, render_configuration_summary, render_detailed_results_header, render_result_group
from gui.backend.exporter import generate_csv_report, generate_pdf_report

def render_history_page():
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    
    batches = sm.list_batches()
    if not batches:
        st.info("No benchmark history found. Start a new run in the 'Benchmark Workspace'.")
        return

    batch_ids = [b["id"] for b in batches]
    
    # Ensure the session state key exists and is valid before rendering the selectbox
    if "selected_history_batch" not in st.session_state or st.session_state.selected_history_batch not in batch_ids:
        st.session_state.selected_history_batch = batch_ids[0] if batch_ids else None

    selected_bid = st.selectbox(
        "Select Benchmark Batch to View & Evaluate",
        batch_ids,
        key="selected_history_batch",
        format_func=lambda bid: get_batch_display_name(bid, sm.base_dir),
        help="Select a batch to view its results, charts, and exports."
    )

    if not selected_bid:
        st.info("Please select a batch from the list above.")
        return

    if True:
        bid = selected_bid
        batch_meta_p = os.path.join(sm.base_dir, bid, "batch_results.json")
        if os.path.exists(batch_meta_p):
            try:
                with open(batch_meta_p, 'r') as f:
                    meta = json.load(f)
                
                # Determine historical TDP
                env = meta.get("environment", {})
                selected_dev = env.get("selected_device", "cpu")
                h_tdp_w = None
                if selected_dev in ["cuda", "mps"]:
                    cuda_devices = env.get("cuda_devices", [])
                    if cuda_devices:
                        h_tdp_w = cuda_devices[0].get("tdp_w")
                    if h_tdp_w is None:
                        h_tdp_w = meta.get("benchmark_settings", {}).get("custom_gpu_tdp")
                else:
                    h_tdp_w = env.get("cpu_tdp_w")
                    if h_tdp_w is None:
                        h_tdp_w = meta.get("benchmark_settings", {}).get("custom_cpu_tdp")

                if h_tdp_w is not None:
                    for g in meta["results"]:
                        for m in g["models"]:
                            for r in m["results"]:
                                try:
                                    rt = r.get("Attribution Runtime (sec)")
                                    if rt is None:
                                        rt = r.get("Runtime (sec)")
                                    if rt is not None and ("Estimated Energy Consumption (kWh)" not in r or r["Estimated Energy Consumption (kWh)"] is None or pd.isna(r["Estimated Energy Consumption (kWh)"])):
                                        r["Estimated Energy Consumption (kWh)"] = round((float(rt) * float(h_tdp_w)) / (3600.0 * 1000.0), 8)
                                except Exception:
                                    pass

                all_h_r = []
                for g in meta["results"]:
                    img_idx = g["img_idx"]
                    for m in g["models"]:
                        for r in m["results"]:
                            r_copy = r.copy()
                            r_copy["Image Index"] = img_idx
                            all_h_r.append(r_copy)
                
                if all_h_r:
                    hdf = normalize_metric_columns(pd.DataFrame(all_h_r))
                    hdf["Model_Size"] = hdf["Model"] + " (" + hdf.get("Resolution", "224x224") + ")"
                    h_runtime_col = metric_col(hdf, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
                    h_memory_col = metric_col(hdf, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
                    hdf = hdf.sort_values(by=["Method", "Resolution"])
                    
                    h_total_time = meta.get("total_execution_time", 0)
                    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
                    
                    # Header row with metadata

                    if h_total_time:
                        st.markdown(format_run_duration(format_time(h_total_time)), unsafe_allow_html=True)
                    st.markdown(format_run_timestamps(meta.get('started_at'), meta.get('completed_at')), unsafe_allow_html=True)
                    st.markdown('<div style="margin-top: 1.0rem;"></div>', unsafe_allow_html=True)
                    
                    # --- EXPORT & NAVIGATION BUTTONS (Above configurations) ---
                    hx1, hx2, hx3, hx4 = st.columns([1, 1, 1.4, 1.6])
                    with hx1:
                        h_csv = os.path.join(sm.base_dir, bid, f"{bid}.csv")
                        if not os.path.exists(h_csv):
                            generate_csv_report(meta["results"], h_csv)
                        if os.path.exists(h_csv):
                            with open(h_csv, "rb") as f:
                                st.download_button("📥 Export CSV", data=f, file_name=f"{bid}.csv", mime="text/csv", key=f"csv_{bid}", use_container_width=True)
                    with hx2:
                        h_pdf = os.path.join(sm.base_dir, bid, f"{bid}.pdf")
                        if not os.path.exists(h_pdf):
                            with st.spinner("Generating PDF..."):
                                generate_pdf_report(bid, meta["results"], meta["methods"], h_pdf, h_total_time, meta.get("environment"))
                        if os.path.exists(h_pdf):
                            with open(h_pdf, "rb") as f:
                                st.download_button("📄 Export PDF", data=f, file_name=f"{bid}.pdf", mime="application/pdf", key=f"pdf_{bid}", use_container_width=True)
                    with hx3:
                        if st.button("Repeat Config", key=f"repeat_{bid}", use_container_width=True):
                            settings = meta.get("benchmark_settings", {})
                            if settings:
                                st.session_state.restore_config = {
                                    "settings": settings,
                                    "environment": meta.get("environment", {}),
                                    "methods_info": meta.get("methods_info", [])
                                }
                                st.session_state.is_finished = False
                                st.session_state.benchmark_running = False
                                st.session_state.benchmark_ready_to_run = False
                                st.session_state.last_run_results = []
                                st.session_state.completed_batch_id = ""
                                st.session_state.last_run_batch_id = ""
                                st.session_state.current_page = "Configure"
                                rerun_app()
                    with hx4:
                        if st.button("Delete Batch", key=f"del_{bid}", use_container_width=True):
                            # Pre-compute next batch selection
                            next_bid = None
                            if bid in batch_ids:
                                idx = batch_ids.index(bid)
                                if idx + 1 < len(batch_ids):
                                    next_bid = batch_ids[idx + 1]
                                elif idx - 1 >= 0:
                                    next_bid = batch_ids[idx - 1]
                            
                            sm.delete_batch(bid)
                            check_and_reset_session_after_delete(bid)
                            st.session_state.selected_history_batch = next_bid
                            st.success(f"Batch {bid} deleted.")
                            st.rerun()
                            
                    st.markdown('<div style="margin-top: 1.0rem;"></div>', unsafe_allow_html=True)
                    with st.expander("⚙️ Environment & Configuration Details", expanded=False, key=f"env_details_{bid}_history"):
                        meta_col1, meta_col_spacer, meta_col2 = st.columns([1.8, 0.2, 2.0])
                        with meta_col1:
                            st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Environment Summary</div>', unsafe_allow_html=True)
                            render_environment_summary(meta.get("environment"))
                        with meta_col2:
                            st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Benchmark Configuration</div>', unsafe_allow_html=True)
                            render_configuration_summary(meta.get("benchmark_settings"), meta.get("results"))
                            
                        # Render parameter mapping table if parameterized methods exist
                        render_parameters_mapping_table(meta.get("methods_info", []))

                    render_analytics_sections(hdf, meta["results"])

                # Per-image results
                st.markdown('<hr style="margin: 1.4rem 0 1.4rem 0; border: none; border-top: 1px solid var(--xai-border);">', unsafe_allow_html=True)
                render_detailed_results_header(f"history_results_{bid}")
                for group in meta["results"]:
                    render_result_group(group, meta["methods"], key_suffix=f"history_{bid}")

            except Exception as e:
                header_left, header_right = st.columns([4, 1.2])
                with header_left:
                    st.error(f"Error reading historical data: {str(e)}")
                    st.exception(e)
                with header_right:
                    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
                    if st.button("Delete Batch", key=f"del_corr_{bid}", use_container_width=True):
                        # Pre-compute next batch selection
                        next_bid = None
                        if bid in batch_ids:
                            idx = batch_ids.index(bid)
                            if idx + 1 < len(batch_ids):
                                next_bid = batch_ids[idx + 1]
                            elif idx - 1 >= 0:
                                next_bid = batch_ids[idx - 1]
                        
                        sm.delete_batch(bid)
                        check_and_reset_session_after_delete(bid)
                        st.session_state.selected_history_batch = next_bid
                        st.success(f"Batch {bid} deleted.")
                        st.rerun()
        else:
            st.info("Loading metadata for this batch...")
    else:
        pass
