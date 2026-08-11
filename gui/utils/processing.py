"""
Data Processing Module
Handles core Pandas DataFrame transformations, column normalization, metric type coercions, and schema standardization for benchmark results.
"""
import pandas as pd
import numpy as np


ATTR_RUNTIME_COL = "Runtime (sec)"
ATTR_MEMORY_COL = "Peak Attribution Memory (MB)"
LEGACY_RUNTIME_COL = "Attribution Runtime (sec)"
LEGACY_MEMORY_COL = "Peak Memory (MB)"

METADATA_COLS = [
    "Warmup Runs", "Memory Runs", "Measured Runs", "Memory Scope",
    "task_id", "task_started_at", "task_completed_at", "status", "error"
]

PRESENTATION_COL_ORDER = [
    "Image Index", "Method", "Model", "Resolution", "Input Size",
    "Deletion AUC", "Insertion AUC", "Sensitivity (Max)", "Gini Index", "Infidelity",
    ATTR_RUNTIME_COL, "Attribution Runtime Std (sec)", "Estimated Energy Consumption (kWh)", 
    ATTR_MEMORY_COL, "Attribution Memory Std (MB)"
]


def metric_col(df, preferred, legacy):
    return preferred if preferred in df.columns else legacy

def normalize_metric_columns(df):
    df = df.copy()
    # Remove duplicate column names if any exist from legacy schemas and guarantee unique row index
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = pd.RangeIndex(len(df))
    
    if ATTR_RUNTIME_COL not in df.columns and LEGACY_RUNTIME_COL in df.columns:
        df[ATTR_RUNTIME_COL] = df[LEGACY_RUNTIME_COL]
    if ATTR_MEMORY_COL not in df.columns and LEGACY_MEMORY_COL in df.columns:
        df[ATTR_MEMORY_COL] = df[LEGACY_MEMORY_COL]
        
    if "Estimated Energy Consumption (kWh)" not in df.columns and "Estimated Energy Consumption (kW)" in df.columns:
        df["Estimated Energy Consumption (kWh)"] = df["Estimated Energy Consumption (kW)"]
    elif "Estimated Energy Consumption (kWh)" in df.columns and "Estimated Energy Consumption (kW)" in df.columns:
        df["Estimated Energy Consumption (kWh)"] = df["Estimated Energy Consumption (kWh)"].combine_first(df["Estimated Energy Consumption (kW)"])
        
    # Coerce metric columns to float64 numeric dtypes so historical JSON string nulls ('-', '.', 'None') convert cleanly to np.nan
    non_numeric_text_cols = {
        "Method", "Model", "Resolution", "Original Resolution", "Prediction", 
        "Status", "Device", "Model Cache", "Timing Scope", "Memory Scope", "_task_id",
        "started_at", "completed_at"
    }
    for col in df.columns:
        if col not in non_numeric_text_cols and not col.startswith("Pareto:") and not col.startswith("Runtime–") and col != "Overall":
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # For timing std columns, replace exactly 0.0 with np.nan because a runtime std of 0.0 is a single-run artifact
    std_cols = [c for c in df.columns if any(k in c for k in ["Runtime Std", "Across Images", "Run Std"])]
    for col in std_cols:
        df[col] = df[col].replace(0.0, np.nan)
        
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
        LEGACY_MEMORY_COL, "Input Size (px)"
    ] + METADATA_COLS
    df = df.drop(columns=[c for c in duplicate_cols if c in df.columns], errors="ignore")
    ordered_cols = [c for c in PRESENTATION_COL_ORDER if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + remaining_cols]
