"""
Exporter Module
Generates high-fidelity ReportLab PDF reports and clean CSV summaries for benchmark sessions.
"""
import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime, timedelta

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

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

PDF_COLUMN_HEADER_MAP = {
    "Input Size (px)": "Resolution",
    "Resolution": "Resolution",
    "Attribution Runtime (sec)": "Runtime (s)",
    "Runtime (sec)": "Runtime (s)",
    "Attribution Runtime Median (sec)": "Runtime (s)",
    "Attribution Runtime Std (sec)": "Runtime Std (s)",
    "Estimated Energy Consumption (kWh)": "Energy (kWh)",
    "Peak Attribution Memory (MB)": "Memory (MB)",
    "Peak Memory (MB)": "Memory (MB)",
    "Attribution Memory Std (MB)": "Memory Std (MB)",
    "Warmup Runs": "Warmups",
    "Memory Runs": "Mem Runs",
    "Measured Runs": "Repeats",
    "Sensitivity (Max)": "Sensitivity",
}


def normalize_metric_columns(df):
    """Ensure consistent runtime and memory column naming across legacy and new batches."""
    df = df.copy()
    if ATTR_RUNTIME_COL not in df.columns and LEGACY_RUNTIME_COL in df.columns:
        df[ATTR_RUNTIME_COL] = df[LEGACY_RUNTIME_COL]
    if ATTR_MEMORY_COL not in df.columns and LEGACY_MEMORY_COL in df.columns:
        df[ATTR_MEMORY_COL] = df[LEGACY_MEMORY_COL]
    return df


def add_input_size_column(df):
    """Derive numeric Input Size (px) column from Resolution if absent."""
    df = df.copy()
    if "Input Size (px)" not in df.columns and "Resolution" in df.columns:
        df["Input Size (px)"] = pd.to_numeric(df["Resolution"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return df


def presentation_df(df):
    """Produce clean, standardized DataFrame sorted according to presentation standards."""
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


def _format_table_cell_value(col_name, val):
    """Format individual metric values cleanly for PDF reporting."""
    if pd.isna(val) or val is None or str(val).strip() in ["", "nan", "None", "."]:
        return "-"
    
    col_lower = str(col_name).lower()
    
    # 1. Resolution / Size / Count / Integer columns
    if any(k in col_lower for k in ["size", "resolution", "px", "repeat", "warmup", "runs", "count", "img_idx"]):
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            return str(val)
            
    # 2. Numeric metric values
    if isinstance(val, (int, float)):
        # Energy consumption
        if "energy" in col_lower or "kwh" in col_lower:
            return f"{val:.6f}" if abs(val) < 0.001 else f"{val:.4f}"
            
        # Memory metrics (1 decimal place)
        if "memory" in col_lower or "mb" in col_lower:
            return f"{val:.1f}"

        # Runtime & Quality metrics (Runtime, Std, AUC, Gini, Infidelity, Sensitivity)
        if any(k in col_lower for k in ["runtime", "sec", "auc", "gini", "infidelity", "sensitivity", "std"]):
            return f"{val:.4f}"

        # Fallback for floats
        return f"{val:.4f}" if isinstance(val, float) and not val.is_integer() else str(int(val))
        
    return str(val)


def _format_date_range(batch_id, benchmark_settings, total_time=0):
    """Produce a human-readable execution date range string for PDF report header."""
    cfg = benchmark_settings or {}
    start_str = cfg.get("started_at") or cfg.get("batch_started_at") or cfg.get("start_time")
    end_str = cfg.get("completed_at") or cfg.get("batch_completed_at") or cfg.get("end_time")
    
    start_dt = None
    end_dt = None

    if start_str:
        try:
            start_dt = datetime.fromisoformat(str(start_str).replace("Z", "+00:00"))
        except Exception:
            pass
            
    if end_str:
        try:
            end_dt = datetime.fromisoformat(str(end_str).replace("Z", "+00:00"))
        except Exception:
            pass

    if not start_dt and batch_id and "Batch_" in batch_id:
        parts = batch_id.split("_")
        if len(parts) >= 3:
            try:
                start_dt = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S")
            except Exception:
                pass

    if start_dt and not end_dt and total_time > 0:
        end_dt = start_dt + timedelta(seconds=float(total_time))

    if start_dt and end_dt:
        if start_dt.date() == end_dt.date():
            return f"{start_dt.strftime('%b %d, %Y %H:%M:%S')} to {end_dt.strftime('%H:%M:%S')}"
        else:
            return f"{start_dt.strftime('%b %d, %Y %H:%M:%S')} to {end_dt.strftime('%b %d, %Y %H:%M:%S')}"
    elif start_dt:
        return start_dt.strftime('%b %d, %Y %H:%M:%S')
    else:
        return datetime.now().strftime('%b %d, %Y')


class NumberedCanvas(canvas.Canvas):
    """Two-pass ReportLab canvas to render page borders and total page count ('Page X of Y')."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header accent bar
        self.setStrokeColor(colors.HexColor("#0d9488"))
        self.setLineWidth(1)
        self.line(36, 565, 806, 565)
        
        # Footer accent bar & page numbers
        self.line(36, 36, 806, 36)
        self.drawString(36, 24, "XAI Efficiency Benchmark Report")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(806, 24, page_text)
        self.restoreState()


def generate_pdf_report(batch_id, results_data, selected_methods, output_path, total_time=0, environment=None, benchmark_settings=None):
    """
    Compiles a polished, multi-page landscape ReportLab PDF report.
    - Page 1: Environment & Benchmark Configuration Metadata + Overall Efficiency Summary Table
    - Page 2: Global Analytical Visualizations (Efficiency Bar Charts, Memory Footprints, Pareto Frontiers)
    - Pages 3+: Per-Image & Model Architecture Evaluation Pages (Side-by-side heatmaps + Metrics Tables)
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(A4),
        leftMargin=36,
        rightMargin=36,
        topMargin=42,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0d9488"),
        alignment=1,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        alignment=1,
        spaceAfter=12
    )
    
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1e293b"),
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        "TableCellLeft",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1e293b"),
        alignment=0
    )

    story = []

    # =========================================================================
    # --- PAGE 1: ENVIRONMENT & CONFIGURATION METADATA SUMMARY ---
    # =========================================================================
    story.append(Paragraph("XAI Efficiency Benchmark — Summary Report", title_style))
    
    time_str = "N/A"
    if total_time:
        h = int(total_time // 3600)
        m = int((total_time % 3600) // 60)
        s = int(total_time % 60)
        time_str = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s" if m > 0 else f"{total_time:.1f}s"
    
    date_range_str = _format_date_range(batch_id, benchmark_settings, total_time)
    story.append(Paragraph(f"Batch ID: {batch_id} &nbsp;|&nbsp; Run Date: {date_range_str} &nbsp;|&nbsp; Total Wall Time: {time_str}", subtitle_style))

    # Build Side-by-Side Metadata Tables (Environment & Settings)
    env = environment or {}
    gpu_list = [f"{d.get('name', 'GPU')} ({d.get('tdp_w')}W TDP)" if d.get('tdp_w') else d.get('name', 'GPU') for d in env.get("cuda_devices", [])]
    gpu_names = ", ".join(gpu_list) or "N/A"
    cpu_display = env.get("processor", "Unknown CPU")
    if env.get("cpu_tdp_w"): cpu_display = f"{cpu_display} ({env.get('cpu_tdp_w')}W TDP)"

    env_data = [
        [Paragraph("Parameter", table_header_style), Paragraph("Details", table_header_style)],
        [Paragraph("Platform / OS", table_cell_style), Paragraph(str(env.get("platform", "Unknown")), table_cell_left)],
        [Paragraph("Python Version", table_cell_style), Paragraph(str(env.get("python_version", "Unknown")), table_cell_left)],
        [Paragraph("PyTorch Version", table_cell_style), Paragraph(str(env.get("torch_version", "Unknown")), table_cell_left)],
        [Paragraph("CUDA Version", table_cell_style), Paragraph(str(env.get("torch_cuda_version") or "N/A"), table_cell_left)],
        [Paragraph("Active Device", table_cell_style), Paragraph(str(env.get("selected_device", "Unknown")), table_cell_left)],
        [Paragraph("CPU Model", table_cell_style), Paragraph(cpu_display, table_cell_left)],
        [Paragraph("GPU Model(s)", table_cell_style), Paragraph(gpu_names, table_cell_left)],
    ]
    t_env = Table(env_data, colWidths=[110, 240])
    t_env.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d9488")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))

    all_r_temp = []
    for g in results_data:
        for m in g.get("models", []): all_r_temp.extend(m.get("results", []))
    fdf_temp = pd.DataFrame(all_r_temp) if all_r_temp else pd.DataFrame()

    cfg = benchmark_settings or {}
    
    # 1. Models list preserving UI selection order
    if cfg.get("models"):
        models_list = [m for m in cfg["models"] if "Model" not in fdf_temp.columns or m in fdf_temp["Model"].values]
    elif "Model" in fdf_temp.columns:
        models_list = list(dict.fromkeys(fdf_temp["Model"]))
    else:
        models_list = []

    # 2. Methods list preserving UI selection order
    cfg_methods = selected_methods or cfg.get("methods") or cfg.get("selected_methods")
    if cfg_methods:
        methods_list = list(dict.fromkeys(cfg_methods))
        if "Method" in fdf_temp.columns:
            methods_list = [m for m in methods_list if any(m.lower() == rm.lower() for rm in fdf_temp["Method"].values)]
    elif "Method" in fdf_temp.columns:
        methods_list = list(dict.fromkeys(fdf_temp["Method"]))
    else:
        methods_list = []

    # 3. Sizes list preserving UI selection order
    if cfg.get("input_sizes"):
        sizes_list = [str(s) for s in cfg["input_sizes"]]
    elif "Input Size (px)" in fdf_temp.columns:
        sizes_list = [str(s) for s in dict.fromkeys(fdf_temp["Input Size (px)"])]
    elif "Resolution" in fdf_temp.columns:
        sizes_list = [str(s) for s in dict.fromkeys(fdf_temp["Resolution"])]
    else:
        sizes_list = []

    # 4. Quality Metrics list and detail formatting
    quality_metrics = cfg.get("selected_quality_metrics") or cfg.get("quality_metrics") or []
    if not quality_metrics and cfg.get("enable_quality_metrics") is False:
        quality_metrics = []
    elif not quality_metrics and not fdf_temp.empty:
        scanned_qm = []
        for qm_name in ["Gini Index", "Deletion AUC", "Insertion AUC", "Infidelity", "Sensitivity (Max)"]:
            if qm_name in fdf_temp.columns and fdf_temp[qm_name].notna().any():
                scanned_qm.append(qm_name)
        quality_metrics = scanned_qm

    qm_count = len(quality_metrics) if quality_metrics else 0
    qm_detail = ", ".join(quality_metrics) if quality_metrics else "None (Disabled)"

    run_order = str(cfg.get("run_order", "Balanced"))
    random_seed = cfg.get("random_seed") if cfg.get("random_seed") is not None else cfg.get("seed")
    order_detail = f"Randomized (Seed: {random_seed})" if run_order == "Randomized" and random_seed is not None else run_order

    cfg_data = [
        [Paragraph("Parameter", table_header_style), Paragraph("Count", table_header_style), Paragraph("Details", table_header_style)],
        [Paragraph("Images Analyzed", table_cell_style), Paragraph(str(len(results_data)), table_cell_style), Paragraph("-", table_cell_left)],
        [Paragraph("Models Selected", table_cell_style), Paragraph(str(len(models_list)), table_cell_style), Paragraph(", ".join(models_list), table_cell_left)],
        [Paragraph("Methods Selected", table_cell_style), Paragraph(str(len(methods_list)), table_cell_style), Paragraph(", ".join(methods_list), table_cell_left)],
        [Paragraph("Input Resolutions", table_cell_style), Paragraph(str(len(sizes_list)), table_cell_style), Paragraph(", ".join(sizes_list), table_cell_left)],
        [Paragraph("Quality Metrics", table_cell_style), Paragraph(str(qm_count), table_cell_style), Paragraph(qm_detail, table_cell_left)],
        [Paragraph("Warmup Runs", table_cell_style), Paragraph("-", table_cell_style), Paragraph(str(cfg.get("warmup_runs", "-")), table_cell_left)],
        [Paragraph("Measured Repeats", table_cell_style), Paragraph("-", table_cell_style), Paragraph(str(cfg.get("repeat_count", "-")), table_cell_left)],
        [Paragraph("Memory Runs", table_cell_style), Paragraph("-", table_cell_style), Paragraph(str(cfg.get("memory_runs", "-")), table_cell_left)],
        [Paragraph("Task Order Strategy", table_cell_style), Paragraph("-", table_cell_style), Paragraph(order_detail, table_cell_left)],
    ]
    t_cfg = Table(cfg_data, colWidths=[110, 50, 190])
    t_cfg.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d9488")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))

    meta_wrapper = Table([
        [Paragraph("Environment Summary", section_heading), Paragraph("Benchmark Configuration", section_heading)],
        [t_env, t_cfg]
    ], colWidths=[360, 360])
    meta_wrapper.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(meta_wrapper)
    story.append(Spacer(1, 10))

    # Method Parameters Table (if parameterized methods exist)
    methods_info = cfg.get("parameterized_methods") or cfg.get("methods_info") or cfg.get("current_batch_methods_info") or cfg.get("expanded_methods") or cfg.get("methods_params") or []
    
    if not methods_info and methods_list:
        try:
            from utils.loader import expand_xai_methods_with_params
            methods_info = expand_xai_methods_with_params(methods_list)
        except Exception:
            pass

    parameterized = [m for m in methods_info if isinstance(m, dict) and m.get("params")]
    
    if parameterized:
        story.append(Paragraph("Method Parameters", section_heading))
        
        param_table_data = [
            [Paragraph("Method Identifier", table_header_style), Paragraph("Base Algorithm", table_header_style), Paragraph("Configured Parameters", table_header_style)]
        ]
        for m in parameterized:
            disp_name = str(m.get("display_name") or m.get("name") or m.get("method") or "-")
            base_name = str(m.get("base_name") or m.get("algorithm") or disp_name).replace("_", " ").title()
            params = m.get("params", {})
            
            formatted_params = []
            if isinstance(params, dict):
                for k, v in params.items():
                    k_pretty = str(k).replace("_", " ").title()
                    if isinstance(v, (list, tuple)):
                        v_str = f"({v[0]}, {v[1]}, {v[2]})" if len(v) == 3 else str(v)
                    else:
                        v_str = str(v)
                    formatted_params.append(f"{k_pretty}: {v_str}")
                params_str = ", ".join(formatted_params)
            else:
                params_str = str(params)

            param_table_data.append([
                Paragraph(disp_name, table_cell_left),
                Paragraph(base_name, table_cell_left),
                Paragraph(params_str, table_cell_left)
            ])
            
        t_param = Table(param_table_data, colWidths=[180, 180, 360])
        t_param.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d9488")),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(t_param)
        story.append(Spacer(1, 10))

    # Performance Evaluation Summary Table
    if not fdf_temp.empty:
        story.append(Paragraph("Overall Performance & Efficiency Evaluation Summary", section_heading))
        
        fdf_norm = normalize_metric_columns(fdf_temp)
        group_cols = ["Model"]
        if "Input Size (px)" in fdf_norm.columns: group_cols.append("Input Size (px)")
        
        agg_dict = {ATTR_RUNTIME_COL: ["mean", "std"]}
        if ATTR_MEMORY_COL in fdf_norm.columns: agg_dict[ATTR_MEMORY_COL] = "mean"
        if "Estimated Energy Consumption (kWh)" in fdf_norm.columns: agg_dict["Estimated Energy Consumption (kWh)"] = "mean"

        perf_df = fdf_norm.groupby(group_cols).agg(agg_dict).reset_index()
        flat_cols = [f"{c[0]} ({c[1]})" if isinstance(c, tuple) and c[1] else (c[0] if isinstance(c, tuple) else c) for c in perf_df.columns]
        perf_df.columns = flat_cols

        headers = [Paragraph(PDF_COLUMN_HEADER_MAP.get(c, c), table_header_style) for c in perf_df.columns]
        rows = [headers]
        for _, row in perf_df.iterrows():
            r_cells = []
            for col_name in perf_df.columns:
                val_str = _format_table_cell_value(col_name, row[col_name])
                r_cells.append(Paragraph(val_str, table_cell_style))
            rows.append(r_cells)

        col_w = 720 / len(perf_df.columns)
        t_perf = Table(rows, colWidths=[col_w] * len(perf_df.columns))
        t_perf.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d9488")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(t_perf)

    # =========================================================================
    # --- PAGE 2: PERFORMANCE BENCHMARK ANALYTICS ---
    # =========================================================================
    created_chart_files = []
    if not fdf_temp.empty:
        story.append(PageBreak())
        story.append(Paragraph("Performance Benchmark Analytics", title_style))
        story.append(Spacer(1, 14))

        try:
            from components.plots import (
                plot_bubble_chart,
                plot_runtime_memory_scatter,
                plot_model_comparison_grouped,
                plot_model_memory_comparison_grouped,
                plot_method_runtime_log,
                plot_method_memory
            )
        except ImportError:
            plot_bubble_chart = None

        temp_dir = tempfile.gettempdir()

        def fig_to_rl_image(fig, filename, width=4.8*inch, height=2.35*inch):
            if fig is None:
                return Paragraph("Chart Not Available", table_cell_style)
            p = os.path.join(temp_dir, filename)
            try:
                fig.savefig(p, bbox_inches='tight', dpi=180)
                plt.close(fig)
                created_chart_files.append(p)
                return RLImage(p, width=width, height=height)
            except Exception as e:
                print(f"[Chart Error] Could not save {filename}: {e}")
                return Paragraph("Chart Not Available", table_cell_style)

        # 1. Efficiency & Memory Footprint Comparison Charts
        fig_comp1 = None
        fig_comp2 = None
        if plot_bubble_chart:
            try:
                fig_comp1 = plot_model_comparison_grouped(fdf_temp, title="Architecture Efficiency Comparison (Runtime)")
            except Exception as e:
                print(f"[Plot Error] plot_model_comparison_grouped: {e}")

            try:
                fig_comp2 = plot_model_memory_comparison_grouped(fdf_temp, title="Architecture Memory Footprint (Peak MB)")
            except Exception as e:
                print(f"[Plot Error] plot_model_memory_comparison_grouped: {e}")

        img_comp1 = fig_to_rl_image(fig_comp1, f"{batch_id}_chart_comp1.png", width=4.8*inch, height=2.4*inch)
        img_comp2 = fig_to_rl_image(fig_comp2, f"{batch_id}_chart_comp2.png", width=4.8*inch, height=2.4*inch)

        t_analytics1 = Table([[img_comp1, img_comp2]], colWidths=[360, 360])
        t_analytics1.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_analytics1)
        story.append(Spacer(1, 10))

        # 2. Pareto Frontier / Trade-off Charts
        fig_p1_chart = None
        fig_p2_chart = None
        if plot_bubble_chart:
            try:
                fig_p1_chart = plot_bubble_chart(fdf_temp)
                if fig_p1_chart is None:
                    fig_p1_chart = plot_method_runtime_log(fdf_temp, title="Attribution Runtime Distribution")
            except Exception as e:
                print(f"[Plot Error] plot_bubble_chart: {e}")

            try:
                fig_p2_chart = plot_runtime_memory_scatter(fdf_temp)
                if fig_p2_chart is None:
                    fig_p2_chart = plot_method_memory(fdf_temp, title="Peak Memory Overhead")
            except Exception as e:
                print(f"[Plot Error] plot_runtime_memory_scatter: {e}")

        img_p1_chart = fig_to_rl_image(fig_p1_chart, f"{batch_id}_chart_pareto1.png", width=4.8*inch, height=2.4*inch)
        img_p2_chart = fig_to_rl_image(fig_p2_chart, f"{batch_id}_chart_pareto2.png", width=4.8*inch, height=2.4*inch)

        t_analytics2 = Table([[img_p1_chart, img_p2_chart]], colWidths=[360, 360])
        t_analytics2.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_analytics2)

    # =========================================================================
    # --- PAGES 3+: PER-IMAGE & ARCHITECTURE COLLAGES ---
    # =========================================================================
    collage_cell_style = ParagraphStyle(
        "CollageCellLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=6.5,
        leading=7.5,
        textColor=colors.HexColor("#0f172a"),
        alignment=1
    )

    for group in results_data:
        img_idx = group.get('img_idx', 1)
        architectures = list(dict.fromkeys([m["model"] for m in group.get("models", []) if "model" in m]))
        
        for arch in architectures:
            arch_models = [m for m in group.get("models", []) if m.get("model") == arch]
            sample_m = arch_models[0]

            pred_parts = []
            for m_data in arch_models:
                sz = m_data.get("input_size") or (m_data["results"][0].get("Input Size (px)") if m_data.get("results") else 0)
                sz_str = f"({sz}px)" if sz else ""
                pred = m_data.get("prediction") or (m_data["results"][0].get("Prediction") if m_data.get("results") else "Unknown")
                pred_parts.append(f"{pred} {sz_str}".strip())
            pred_display = ", ".join(pred_parts) if pred_parts else "N/A"

            story.append(PageBreak())
            story.append(Paragraph(f"Image {img_idx} &nbsp;|&nbsp; Architecture: {arch} &nbsp;|&nbsp; Prediction: {pred_display}", section_heading))
            story.append(Spacer(1, 6))

            # Left Column: Input Image
            img_path = os.path.join(sample_m.get("session_dir", ""), "input_image.jpg")
            img_element = Paragraph("Input Image N/A", table_cell_style)
            if os.path.exists(img_path):
                img_element = RLImage(img_path, width=1.6*inch, height=1.6*inch)

            # Right Column: Heatmap Collage Sub-table
            collage_rows = []
            for method in selected_methods:
                method_row = []
                for m_data in arch_models:
                    res = next((r for r in m_data.get("results", []) if r.get("Method", "").lower() == method.lower()), None)
                    cell_content = Paragraph("-", table_cell_style)
                    if res:
                        h_p = os.path.join(m_data.get("session_dir", ""), "heatmaps", f"{res['Method']}.png")
                        if os.path.exists(h_p):
                            lbl_text = f"<b>{res['Method']}</b> ({m_data.get('input_size', '')}px)"
                            lbl_p = Paragraph(lbl_text, collage_cell_style)
                            img_p = RLImage(h_p, width=0.85*inch, height=0.85*inch)
                            cell_tbl = Table([[lbl_p], [img_p]], colWidths=[0.9*inch])
                            cell_tbl.setStyle(TableStyle([
                                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                                ('TOPPADDING', (0, 0), (-1, -1), 0),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
                            ]))
                            cell_content = cell_tbl
                    method_row.append(cell_content)
                collage_rows.append(method_row)

            t_collage = Table(collage_rows)
            t_collage.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 2),
                ('RIGHTPADDING', (0, 0), (-1, -1), 2),
            ]))

            vis_wrapper = Table([[img_element, t_collage]], colWidths=[150, 570])
            vis_wrapper.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            story.append(vis_wrapper)
            story.append(Spacer(1, 10))

            # Consolidated Metrics Table at Bottom
            arch_results = [r for method in selected_methods for m in arch_models for r in [next((x for x in m.get("results", []) if x.get("Method", "").lower() == method.lower()), None)] if r]
            if arch_results:
                df = presentation_df(pd.DataFrame(arch_results))
                raw_cols = [c for c in ["Method", "Input Size (px)", ATTR_RUNTIME_COL, "Attribution Runtime Std (sec)", "Estimated Energy Consumption (kWh)", ATTR_MEMORY_COL, "Attribution Memory Std (MB)"] if c in df.columns]
                for qm in ["Gini Index", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)", "Infidelity", "Status"]:
                    if qm in df.columns and any(df[qm].notna()): raw_cols.append(qm)

                headers = [Paragraph(PDF_COLUMN_HEADER_MAP.get(c, c), table_header_style) for c in raw_cols]
                tbl_rows = [headers]
                for _, r_data in df[raw_cols].iterrows():
                    r_cells = []
                    for c in raw_cols:
                        val_str = _format_table_cell_value(c, r_data[c])
                        r_cells.append(Paragraph(val_str, table_cell_style))
                    tbl_rows.append(r_cells)

                c_width = 720 / len(raw_cols)
                t_metrics = Table(tbl_rows, colWidths=[c_width] * len(raw_cols))
                t_metrics.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d9488")),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ]))
                story.append(t_metrics)

    try:
        doc.build(story, canvasmaker=NumberedCanvas)
    finally:
        for tmp_f in created_chart_files:
            try:
                if os.path.exists(tmp_f):
                    os.remove(tmp_f)
            except Exception:
                pass
    return True


def generate_csv_report(results_data, output_path):
    """Compiles and exports the complete flat presentation DataFrame to CSV format."""
    all_r = []
    for g in results_data:
        for m in g.get("models", []):
            all_r.extend(m.get("results", []))
    if all_r:
        presentation_df(pd.DataFrame(all_r)).to_csv(output_path, index=False)
        return True
    return False
