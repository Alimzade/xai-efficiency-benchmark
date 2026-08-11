"""
Active Run Page Module
Renders the live benchmarking execution page, including real-time progress, live metrics, and active task queues.
"""
import os
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import sm, default_device_mode
from utils.processing import normalize_metric_columns
from backend.benchmark_runner import collect_environment_metadata
from backend.exporter import generate_csv_report, generate_pdf_report
from utils.helpers import get_device_string, sorted_result_groups, format_time, timestamp_now, format_run_timestamps, format_run_duration, parse_input_sizes, rerun_app
from components.cards import render_live_elapsed_timer, render_analytics_sections, render_environment_summary, render_parameters_mapping_table, render_configuration_summary, render_detailed_results_header, render_result_group


def render_active_run_page():
    title_text = "⚡ Benchmark Execution Engine"
    if st.session_state.is_finished:
        title_text += " | Results"
        
    st.markdown(f"""
        <div style="text-align: center; margin-top: 0.8rem; margin-bottom: -2.5rem !important;">
            <h3 style="background: linear-gradient(90deg, #60a5fa, #2dd4bf); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.55rem; font-weight: 700; margin: 0; display: inline-block; letter-spacing: -0.02em;">
                {title_text}
            </h3>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.benchmark_ready_to_run:
        st.info("Preparing benchmark environment... Starting shortly.")
        components.html("""
            <script>
                const targetLabel = "Auto-start benchmark engine";
                let clicked = false;

                function hideAndClick(button) {
                    if (clicked || !button || button.disabled) {
                        return;
                    }
                    clicked = true;
                    const wrapper = button.closest('[data-testid="stButton"]') || button.parentElement;
                    if (wrapper) {
                        wrapper.style.display = "none";
                        wrapper.style.height = "0";
                        wrapper.style.overflow = "hidden";
                    }
                    setTimeout(() => button.click(), 1500);
                }

                function findButton() {
                    const buttons = Array.from(window.parent.document.querySelectorAll("button"));
                    return buttons.find((button) => button.textContent.trim() === targetLabel);
                }

                function scan() {
                    try {
                        hideAndClick(findButton());
                    } catch (error) {
                        clicked = true;
                    }
                }

                const observer = new MutationObserver(scan);
                observer.observe(window.parent.document.body, { childList: true, subtree: true });
                scan();
                setTimeout(() => observer.disconnect(), 5000);
            </script>
        """, height=0)
        
        if st.button("Auto-start benchmark engine", type="primary", use_container_width=True):
            if not st.session_state.get("batch_started_at"):
                st.session_state.batch_started_at = timestamp_now()
                cfg = sm.load_batch_config(st.session_state.current_batch_id)
                if cfg:
                    cfg["started_at"] = st.session_state.batch_started_at
                    sm.save_batch_config(st.session_state.current_batch_id, cfg)
            st.session_state.batch_start_time = time.time()
            st.session_state.batch_completed_at = ""
            st.session_state.total_execution_time = 0
            st.session_state.benchmark_ready_to_run = False
            st.session_state.benchmark_running = True
            st.session_state.first_run_render = True
            rerun_app()
            
    elif st.session_state.benchmark_running and not st.session_state.is_finished:
        task_queue = st.session_state.task_queue
        total_steps = len(task_queue)
        idx = st.session_state.run_progress_idx
        current_task = task_queue[idx] if idx < total_steps else {}
        cur_mod = current_task.get("model_name", "?")
        cur_met = current_task.get("method_name", "?")
        cur_size = current_task.get("target_size", "?")
        img_i = current_task.get("img_i", 0)

        # Render the summary bar of the active run configuration at the top (visually stable anchor)
        st.markdown(f"""
            <div class="run-summary-bar" style="margin-top: 5px; margin-bottom: 8px;">
                <span class="run-summary-item">Models <strong>{len(st.session_state.current_batch_models)}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Size variations <strong>{len(st.session_state.current_batch_sizes)}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Methods <strong>{len(st.session_state.current_batch_methods)}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Images <strong>{len(st.session_state.prepared_img_sources)}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Repeats/config <strong>{st.session_state.current_repeats}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Warmups/config <strong>{st.session_state.current_warmups}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Memory runs/config <strong>{st.session_state.current_memory_runs}</strong></span>
            </div>
            """, unsafe_allow_html=True)

        # Progress bar (loading line)
        st.progress(idx / total_steps if total_steps else 0)
        
        # Step status details and live elapsed timer below the loading line
        status_col, timer_col = st.columns([5, 1])
        with status_col:
            st.markdown(
                f"<div class='status-pulse'>STEP {idx + 1}/{total_steps}: {cur_met} on {cur_mod} @ {cur_size}px (Image {img_i + 1})</div>",
                unsafe_allow_html=True
            )
        with timer_col:
            render_live_elapsed_timer(st.session_state.batch_start_time)
            
        st.divider()
        
        if st.button("🛑 Stop Benchmark", use_container_width=True):
            st.session_state.stop_requested = True
            st.session_state.benchmark_running = False
            st.session_state.last_run_results = []
            st.session_state.task_queue = []
            st.session_state.prepared_img_sources = []
            st.session_state.last_run_batch_id = ""
            st.session_state.completed_batch_id = ""
            st.session_state.benchmark_ready_to_run = False
            st.session_state.current_batch_methods = []
            st.session_state.current_batch_models = []
            st.session_state.current_batch_sizes = []
            st.session_state.current_page = "Configure"
            rerun_app()
            
        st.markdown('<div style="margin-top: 1.0rem;"></div>', unsafe_allow_html=True)
        with st.expander("⚙️ Environment & Configuration Details", expanded=True, key=f"env_details_{st.session_state.current_batch_id}_active"):
            meta_col1, meta_col_spacer, meta_col2 = st.columns([1.8, 0.2, 2.0])
            with meta_col1:
                st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Environment Summary</div>', unsafe_allow_html=True)
                render_environment_summary(collect_environment_metadata(
                    get_device_string(st.session_state.current_device_mode),
                    custom_cpu_tdp=st.session_state.get("current_cpu_tdp"),
                    custom_gpu_tdp=st.session_state.get("current_gpu_tdp")
                ))
            with meta_col2:
                st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Benchmark Configuration</div>', unsafe_allow_html=True)
                active_settings = {
                    "models": st.session_state.current_batch_models,
                    "methods": st.session_state.current_batch_methods,
                    "input_sizes": st.session_state.current_batch_sizes,
                    "repeat_count": st.session_state.current_repeats,
                    "warmup_runs": st.session_state.current_warmups,
                    "memory_runs": st.session_state.current_memory_runs,
                    "selected_quality_metrics": st.session_state.current_selected_quality_metrics if st.session_state.get("current_enable_quality_metrics", False) else []
                }
                render_configuration_summary(active_settings, st.session_state.last_run_results)
            
            # Render parameter mapping table if parameterized methods exist
            render_parameters_mapping_table(st.session_state.get("current_batch_methods_info", []))
            
        if st.session_state.last_run_results:
            render_detailed_results_header("current_live_results", "Completed Results So Far")
            for group in sorted_result_groups(st.session_state.last_run_results):
                render_result_group(group, st.session_state.current_batch_methods, key_suffix=f"live_{st.session_state.current_batch_id}")
                
        if st.session_state.get("first_run_render"):
            components.html("""
                <script>
                    const targetLabel = "Trigger engine step";
                    function scanAndHide() {
                        const buttons = Array.from(window.parent.document.querySelectorAll("button"));
                        const btn = buttons.find((button) => button.textContent.trim() === targetLabel);
                        if (btn) {
                            const wrapper = btn.closest('[data-testid="stElementContainer"]') || btn.closest('[data-testid="stButton"]') || btn.parentElement;
                            if (wrapper) {
                                wrapper.style.display = "none";
                            }
                            // Wait 600ms before clicking to let the page fully render
                            setTimeout(() => {
                                btn.click();
                            }, 600);
                            return true;
                        }
                        return false;
                    }
                    // Poll to hide it as early as possible
                    let interval = setInterval(function() {
                        if (scanAndHide()) {
                            clearInterval(interval);
                        }
                    }, 50);
                    // Disconnect after 2 seconds just in case
                    setTimeout(function() { clearInterval(interval); }, 2000);
                </script>
            """, height=0)
            st.markdown('<div style="display:none;">', unsafe_allow_html=True)
            if st.button("Trigger engine step", key="trigger_engine_step_btn", use_container_width=True):
                st.session_state.first_run_render = False
                rerun_app()
            st.markdown('</div>', unsafe_allow_html=True)
            
    elif st.session_state.is_finished:

        if st.session_state.completion_notice_batch_id != st.session_state.current_batch_id:
            st.toast("Benchmark complete. Results are ready.", icon="✅")
            st.session_state.completion_notice_batch_id = st.session_state.current_batch_id

        # Determine live TDP
        dev_mode = get_device_string(st.session_state.current_device_mode)
        live_tdp_w = st.session_state.get("current_gpu_tdp") if dev_mode in ["cuda", "mps"] else st.session_state.get("current_cpu_tdp")

        if live_tdp_w is not None:
            for g in sorted_result_groups(st.session_state.last_run_results):
                for m in g.get("models", []):
                    for r in m.get("results", []):
                        try:
                            rt = r.get("Attribution Runtime (sec)")
                            if rt is None:
                                rt = r.get("Runtime (sec)")
                            if rt is not None and ("Estimated Energy Consumption (kWh)" not in r or r["Estimated Energy Consumption (kWh)"] is None or pd.isna(r["Estimated Energy Consumption (kWh)"])):
                                r["Estimated Energy Consumption (kWh)"] = round((float(rt) * float(live_tdp_w)) / (3600.0 * 1000.0), 8)
                        except Exception:
                            pass

        all_r = []
        for g in sorted_result_groups(st.session_state.last_run_results):
            img_idx = g.get("img_idx")
            if img_idx is None:
                img_id_str = str(g.get("image_id", "Image 1"))
                parts = img_id_str.split()
                img_idx = int(parts[1]) - 1 if len(parts) > 1 and parts[1].isdigit() else 0
            for m in g.get("models", []):
                for r in m.get("results", []):
                    r_copy = r.copy()
                    r_copy["Image Index"] = img_idx
                    all_r.append(r_copy)
            
        if all_r:
            fdf = normalize_metric_columns(pd.DataFrame(all_r))
            result_groups = sorted_result_groups(st.session_state.last_run_results)
            

            st.markdown('<div style="margin-top: 0.8rem;"></div>', unsafe_allow_html=True)
            if st.session_state.total_execution_time:
                st.markdown(format_run_duration(format_time(st.session_state.total_execution_time)), unsafe_allow_html=True)
            st.markdown(format_run_timestamps(st.session_state.batch_started_at, st.session_state.batch_completed_at), unsafe_allow_html=True)
            st.markdown('<div style="margin-top: 1.0rem;"></div>', unsafe_allow_html=True)
            
            # --- EXPORT & NAVIGATION BUTTONS (Above configurations) ---
            ex1, ex2, ex3, ex4 = st.columns([1, 1, 1.4, 1.6])
            with ex1:
                if st.session_state.current_batch_id and os.path.exists(os.path.join(sm.base_dir, st.session_state.current_batch_id)):
                    csv_path = os.path.join(sm.base_dir, st.session_state.current_batch_id, f"{st.session_state.current_batch_id}.csv")
                    if not os.path.exists(csv_path):
                        generate_csv_report(st.session_state.last_run_results, csv_path)
                    if os.path.exists(csv_path):
                        with open(csv_path, "rb") as f:
                            st.download_button("📥 Export CSV", data=f, file_name=f"{st.session_state.current_batch_id}.csv", mime="text/csv", use_container_width=True)
            with ex2:
                if st.session_state.current_batch_id and os.path.exists(os.path.join(sm.base_dir, st.session_state.current_batch_id)):
                    pdf_path = os.path.join(sm.base_dir, st.session_state.current_batch_id, f"{st.session_state.current_batch_id}.pdf")
                    alt_pdf = os.path.join(sm.base_dir, st.session_state.current_batch_id, "report.pdf")
                    target_pdf = pdf_path if os.path.exists(pdf_path) else (alt_pdf if os.path.exists(alt_pdf) else pdf_path)
                    if not os.path.exists(target_pdf):
                        active_cfg = {
                            "models": st.session_state.current_batch_models,
                            "methods": st.session_state.current_batch_methods,
                            "parameterized_methods": st.session_state.get("current_batch_methods_info", []),
                            "input_sizes": st.session_state.current_batch_sizes,
                            "repeat_count": st.session_state.current_repeats,
                            "warmup_runs": st.session_state.current_warmups,
                            "memory_runs": st.session_state.current_memory_runs,
                            "run_order": st.session_state.current_run_order,
                            "random_seed": st.session_state.get("current_random_seed", 42),
                            "enable_quality_metrics": st.session_state.get("current_enable_quality_metrics", False),
                            "selected_quality_metrics": st.session_state.get("current_selected_quality_metrics", []),
                        }
                        generate_pdf_report(
                             st.session_state.current_batch_id,
                             st.session_state.last_run_results,
                             st.session_state.current_batch_methods,
                             target_pdf,
                             st.session_state.total_execution_time,
                             collect_environment_metadata(
                                 get_device_string(st.session_state.current_device_mode),
                                 custom_cpu_tdp=st.session_state.get("current_cpu_tdp"),
                                 custom_gpu_tdp=st.session_state.get("current_gpu_tdp")
                             ),
                             benchmark_settings=active_cfg
                        )
                    if os.path.exists(target_pdf):
                        with open(target_pdf, "rb") as f:
                            st.download_button("📄 Export PDF", data=f, file_name=f"{st.session_state.current_batch_id}.pdf", mime="application/pdf", use_container_width=True)
            with ex3:
                if st.button("Repeat Config", key="repeat_current_active_btn", help="Load this configuration back into your workspace inputs to tweak or run it again.", use_container_width=True):
                    st.session_state.restore_config = {
                        "settings": {
                            "models": st.session_state.current_batch_models if st.session_state.current_batch_models else st.session_state.selected_models,
                            "methods": st.session_state.current_batch_methods if st.session_state.current_batch_methods else st.session_state.selected_methods,
                            "selected_methods": st.session_state.sh_methods if st.session_state.get("sh_methods") else st.session_state.selected_methods,
                            "methods_info": st.session_state.get("current_batch_methods_info", []),
                            "input_sizes": st.session_state.current_batch_sizes if st.session_state.current_batch_sizes else parse_input_sizes(st.session_state.get("input_size_str", "224")),
                            "repeat_count": st.session_state.current_repeats,
                            "warmup_runs": st.session_state.current_warmups,
                            "memory_runs": st.session_state.current_memory_runs,
                            "run_order": st.session_state.current_run_order,
                            "random_seed": st.session_state.get("current_random_seed", 42),
                            "enable_quality_metrics": st.session_state.current_enable_quality_metrics,
                            "selected_quality_metrics": st.session_state.current_selected_quality_metrics,
                            "xai_params": st.session_state.get("sh_xai_params", {})
                        },
                        "environment": {
                            "selected_device": st.session_state.current_device_mode
                        }
                    }
                    st.session_state.is_finished = False
                    st.session_state.benchmark_running = False
                    st.session_state.benchmark_ready_to_run = False
                    st.session_state.last_run_results = []
                    st.session_state.completed_batch_id = ""
                    st.session_state.last_run_batch_id = ""
                    st.session_state.current_page = "Configure"
                    rerun_app()
            with ex4:
                if st.button("⬅️ Setup Another Run", key="back_from_run_btn", help="Reset all configuration inputs back to defaults to start a fresh benchmark from scratch.", use_container_width=True):
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
                    st.session_state.current_page = "Configure"
                    st.session_state.selected_models = ["resnet50"]
                    st.session_state.selected_methods = ["Saliency", "Integrated_Gradients"]
                    st.session_state.input_size_str = "224"
                    st.session_state.selected_repeats = 5
                    st.session_state.selected_warmups = 3
                    st.session_state.selected_run_order = "Balanced"
                    st.session_state.selected_device_mode = default_device_mode
                    st.session_state.last_run_results = []
                    st.session_state.completed_batch_id = ""
                    st.session_state.last_run_batch_id = ""
                    st.session_state.current_batch_methods = []
                    st.session_state.current_batch_models = []
                    st.session_state.current_batch_sizes = []
                    rerun_app()
                    
            st.markdown('<div style="margin-top: 1.0rem;"></div>', unsafe_allow_html=True)
            with st.expander("⚙️ Environment & Configuration Details", expanded=True, key=f"env_details_{st.session_state.current_batch_id}_finished"):
                meta_col1, meta_col_spacer, meta_col2 = st.columns([1.8, 0.2, 2.0])
                with meta_col1:
                    st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Environment Summary</div>', unsafe_allow_html=True)
                    render_environment_summary(collect_environment_metadata(
                        get_device_string(st.session_state.current_device_mode),
                        custom_cpu_tdp=st.session_state.get("current_cpu_tdp"),
                        custom_gpu_tdp=st.session_state.get("current_gpu_tdp")
                    ))
                with meta_col2:
                    st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Benchmark Configuration</div>', unsafe_allow_html=True)
                    active_settings = {
                        "models": st.session_state.current_batch_models,
                        "methods": st.session_state.current_batch_methods,
                        "input_sizes": st.session_state.current_batch_sizes,
                        "repeat_count": st.session_state.current_repeats,
                        "warmup_runs": st.session_state.current_warmups,
                        "memory_runs": st.session_state.current_memory_runs,
                        "selected_quality_metrics": st.session_state.selected_quality_metrics if st.session_state.get("enable_quality_metrics", False) else []
                    }
                    render_configuration_summary(active_settings, st.session_state.last_run_results)
                
                # Render parameter mapping table if parameterized methods exist
                render_parameters_mapping_table(st.session_state.get("current_batch_methods_info", []))

            render_analytics_sections(fdf, result_groups)

        st.markdown('<hr style="margin: 1.4rem 0 1.4rem 0; border: none; border-top: 1px solid var(--xai-border);">', unsafe_allow_html=True)
        render_detailed_results_header("current_final_results")
        for group in sorted_result_groups(st.session_state.last_run_results):
            render_result_group(group, st.session_state.current_batch_methods, key_suffix=f"finished_{st.session_state.current_batch_id}")
    else:
        st.info("No active benchmark run. Go to the **Configure Benchmark** page to set up and launch a run!")
