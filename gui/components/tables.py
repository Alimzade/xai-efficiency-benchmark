"""
Table Components Module
Contains helper functions for styling and rendering Pandas DataFrames beautifully in Streamlit.
"""
import pandas as pd

from config import logger
from utils.processing import ATTR_RUNTIME_COL, ATTR_MEMORY_COL, normalize_metric_columns

def style_dataframe(df, raw_precision=False):
    df = normalize_metric_columns(df.copy())
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = pd.RangeIndex(len(df))
    
    # Identify which columns to style with background gradient.
    possible_style_cols = [
        ATTR_RUNTIME_COL, ATTR_MEMORY_COL,
        "Attribution Runtime Median (sec)", "Attribution Runtime Mean (sec)",
        "Attribution Runtime Std (sec)", "Attribution Runtime Min (sec)",
        "Attribution Runtime Max (sec)", "Mean Attribution Runtime (sec)",
        "Std Across Images (sec)", "Repeat Run Std (sec)",
        "Mean Peak Attribution Memory (MB)", "Peak Memory (MB)", "Peak Attribution Memory (MB)",
        "Std Peak Memory (MB)", "Peak Memory Std (MB)", "Attribution Memory Std (MB)", "Repeat Memory Std (MB)",
        "Estimated Energy Consumption (kWh)", "Mean Estimated Energy Consumption (kWh)",
        "Gini Index", "Mean Gini Index",
        "Deletion AUC", "Mean Deletion AUC",
        "Insertion AUC", "Mean Insertion AUC",
        "Infidelity", "Mean Infidelity",
        "Sensitivity (Max)", "Mean Sensitivity (Max)"
    ]
    
    subset_cols = [
        c for c in possible_style_cols 
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().shape[0] > 0
    ]
            
    def make_formatter(col_name):
        def _fmt(val):
            if pd.isna(val) or val is None or val in ["N/A", "-", "None", ""]:
                return "–"
            try:
                v = float(val)
            except Exception:
                return str(val)

            if raw_precision:
                if col_name == "Resolution":
                    return f"{int(round(v))} x {int(round(v))}"
                elif col_name in ["Input Size (px)", "Samples", "Methods", "Resolutions", "Images", "Repeats", "Total Attribution Runs"]:
                    return f"{int(round(v))}"
                elif "kWh" in col_name or "Energy" in col_name:
                    return f"{v:.8f}".rstrip('0').rstrip('.')
                s = f"{v:.6f}".rstrip('0').rstrip('.')
                return s if s else "0"
            else:
                if "MB" in col_name or col_name in [ATTR_MEMORY_COL, "Mean Peak Attribution Memory (MB)", "Std Peak Memory (MB)", "Peak Memory Std (MB)", "Attribution Memory Std (MB)"]:
                    return f"{v:.2f}"
                elif "kWh" in col_name or "Energy" in col_name:
                    return f"{v:.8f}".rstrip('0').rstrip('.')
                elif "sec" in col_name or "Infidelity" in col_name or "Sensitivity" in col_name or col_name in ["Gini Index", "Mean Gini Index", "Deletion AUC", "Mean Deletion AUC", "Insertion AUC", "Mean Insertion AUC"]:
                    return f"{v:.4f}"
                elif col_name == "Resolution":
                    return f"{int(round(v))} x {int(round(v))}"
                elif col_name in ["Input Size (px)", "Samples", "Methods", "Resolutions", "Images", "Repeats", "Total Attribution Runs"]:
                    return f"{int(round(v))}"
                else:
                    return f"{v:.4f}"
        return _fmt

    format_cols = [c for c in df.columns if c not in ["Method", "Model", "Prediction", "Status", "Device", "Model_Size", "Model Cache", "Timing Scope", "Memory Scope"]]
    formatters = {c: make_formatter(c) for c in format_cols}

    styler = df.style
    if subset_cols:
        lower_is_better_cols = [c for c in subset_cols if not any(k in c for k in ["Gini", "Insertion"])]
        higher_is_better_cols = [c for c in subset_cols if any(k in c for k in ["Gini", "Insertion"])]

        try:
            for col in lower_is_better_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    styler = styler.background_gradient(cmap="coolwarm", subset=[col])
            for col in higher_is_better_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    styler = styler.background_gradient(cmap="coolwarm_r", subset=[col])
        except Exception as e:
            logger.warning(f"Background gradient styling skipped due to Pandas Styler incompatibility: {e}")
        
    # Right-align all columns containing metrics or numeric dimensions to keep dashes aligned with numbers
    right_align_cols = [c for c in df.columns if c not in ["Method", "Model", "Resolution", "Prediction", "Status", "Device", "Model_Size"]]
    if right_align_cols:
        styler = styler.set_properties(**{"text-align": "right !important"}, subset=right_align_cols)
        
        # Also right-align the column headers (th) for these columns to align with data cells
        header_styles = []
        for idx, col in enumerate(df.columns):
            if col in right_align_cols:
                header_styles.append({
                    "selector": f"th.col{idx}",
                    "props": [("text-align", "right !important")]
                })
        if header_styles:
            styler = styler.set_table_styles(header_styles, overwrite=False)
        
    styler = styler.format(formatters, na_rep="–")
    if hasattr(styler, "hide"):
        styler = styler.hide()
    elif hasattr(styler, "hide_index"):
        styler = styler.hide_index()
    return styler
