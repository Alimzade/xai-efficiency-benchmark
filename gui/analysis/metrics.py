"""
Metrics Analysis Module
Contains mathematical logic for summarizing metrics, extracting fastest/slowest runs, and aggregating image size scaling behaviors.
"""
import pandas as pd

from gui.utils.processing import add_input_size_column, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL, ATTR_MEMORY_COL, LEGACY_MEMORY_COL, metric_col, normalize_metric_columns, presentation_df

def image_size_summary(df):
    df = add_input_size_column(normalize_metric_columns(df))
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    if "Input Size (px)" not in df.columns or df["Input Size (px)"].nunique() < 2:
        return pd.DataFrame()

    agg_ops = {
        "Mean Attribution Runtime (sec)": (runtime_col, "mean"),
        "Attribution Runtime Std (sec)": (runtime_col, "std"),
    }
    if "Estimated Energy Consumption (kWh)" in df.columns and df["Estimated Energy Consumption (kWh)"].notna().any():
        agg_ops["Mean Estimated Energy Consumption (kWh)"] = ("Estimated Energy Consumption (kWh)", "mean")
    agg_ops["Mean Peak Attribution Memory (MB)"] = (memory_col, "mean")
    agg_ops["Peak Memory Std (MB)"] = (memory_col, "std")
    agg_ops["Samples"] = (runtime_col, "count")

    summary = df.groupby(["Input Size (px)"]).agg(**agg_ops).reset_index()
    # Rename Column to Resolution
    summary = summary.rename(columns={"Input Size (px)": "Resolution"})
    return summary.sort_values("Resolution")

def method_detail_summary(df):
    df = add_input_size_column(normalize_metric_columns(df))
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    
    n_images = int(df["Image Index"].nunique()) if "Image Index" in df.columns else 1
    std_runtime_src = "Attribution Runtime Std (sec)" if (n_images == 1 and "Attribution Runtime Std (sec)" in df.columns) else runtime_col
    std_runtime_func = "mean" if n_images == 1 else "std"

    std_mem_src = "Attribution Memory Std (MB)" if (n_images == 1 and "Attribution Memory Std (MB)" in df.columns) else memory_col
    std_mem_func = "mean" if n_images == 1 else "std"

    agg_dict = {
        "Mean Attribution Runtime (sec)": (runtime_col, "mean"),
        "Attribution Runtime Std (sec)": (std_runtime_src, std_runtime_func),
    }
    if "Estimated Energy Consumption (kWh)" in df.columns and df["Estimated Energy Consumption (kWh)"].notna().any():
        agg_dict["Mean Estimated Energy Consumption (kWh)"] = ("Estimated Energy Consumption (kWh)", "mean")
    agg_dict["Mean Peak Attribution Memory (MB)"] = (memory_col, "mean")
    agg_dict["Peak Memory Std (MB)"] = (std_mem_src, std_mem_func)
    if "Gini Index" in df.columns and df["Gini Index"].notna().any():
        agg_dict["Mean Gini Index"] = ("Gini Index", "mean")
    if "Deletion AUC" in df.columns and df["Deletion AUC"].notna().any():
        agg_dict["Mean Deletion AUC"] = ("Deletion AUC", "mean")
    if "Insertion AUC" in df.columns and df["Insertion AUC"].notna().any():
        agg_dict["Mean Insertion AUC"] = ("Insertion AUC", "mean")
    if "Sensitivity (Max)" in df.columns and df["Sensitivity (Max)"].notna().any():
        agg_dict["Mean Sensitivity (Max)"] = ("Sensitivity (Max)", "mean")
    if "Infidelity" in df.columns and df["Infidelity"].notna().any():
        agg_dict["Mean Infidelity"] = ("Infidelity", "mean")
    agg_dict["Samples"] = (runtime_col, "count")
    summary = df.groupby("Method", sort=False).agg(**agg_dict).reset_index()
    return summary

def fastest_slowest_rows(df, count=5):
    df = presentation_df(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    cols = [c for c in ["Method", "Model", "Resolution", "Prediction", runtime_col, ATTR_MEMORY_COL, "Gini Index"] if c in df.columns]
    fastest = df.nsmallest(count, runtime_col)[cols]
    slowest = df.nlargest(count, runtime_col)[cols]
    return fastest, slowest
