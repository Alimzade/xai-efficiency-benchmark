"""
Plotting Components Module
Contains all Matplotlib and Seaborn charting functions used throughout the application to visualize benchmarking data.
"""
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from gui.data.processing import ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL, ATTR_MEMORY_COL, LEGACY_MEMORY_COL, metric_col, normalize_metric_columns

def plot_bubble_chart(df):
    df = normalize_metric_columns(df)
    
    # We need Runtime, y_metric, Memory, and Gini.
    x_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    y_col = "Insertion AUC" if ("Insertion AUC" in df.columns and df["Insertion AUC"].notna().any()) else "Deletion AUC"
    if y_col not in df.columns or df[y_col].isna().all():
        return None
        
    size_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    color_col = "Gini Index"
    
    req_cols = [x_col, y_col, size_col]
    if color_col in df.columns:
        req_cols.append(color_col)
        
    df_mean = df.groupby("Method")[req_cols].mean().dropna(subset=[x_col, y_col, size_col]).reset_index()
    if df_mean.empty:
        return None
        
    # Match width to the radar chart
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    
    # Scale sizes: normalize to range [50, 500] for visual clarity
    min_sz = df_mean[size_col].min()
    max_sz = df_mean[size_col].max()
    if max_sz == min_sz:
        sizes = [150] * len(df_mean)
    else:
        sizes = 50 + 450 * (df_mean[size_col] - min_sz) / (max_sz - min_sz)
        
    # Use color mapping if Gini is available
    if color_col in df_mean.columns and not df_mean[color_col].isna().all():
        scatter = ax.scatter(
            x=df_mean[x_col], 
            y=df_mean[y_col], 
            s=sizes, 
            c=df_mean[color_col], 
            cmap="viridis", 
            alpha=0.7, 
            edgecolors="w", 
            linewidth=1.5
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f"Sparsity ({color_col})", rotation=270, labelpad=15)
    else:
        scatter = ax.scatter(
            x=df_mean[x_col], 
            y=df_mean[y_col], 
            s=sizes, 
            alpha=0.7, 
            edgecolors="w", 
            linewidth=1.5,
            color="royalblue"
        )
        
    for i, row in df_mean.iterrows():
        ax.annotate(row["Method"], (row[x_col], row[y_col]), xytext=(0, 0), textcoords='offset points', 
                    ha='center', va='center', fontsize=8, fontweight='bold', color='black',
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, lw=0))
                    
    ax.set_xlabel(f"{x_col}\n(Lower is Better)", fontsize=9)
    y_is_lower = "Deletion" in y_col
    ax.set_ylabel(f"{y_col}\n({'Lower' if y_is_lower else 'Higher'} is Better)", fontsize=9)
    ax.set_title("Efficiency vs Quality Trade-offs\n(Bubble Size = Peak Memory)", fontweight='bold', fontsize=11)
    
    # Add an invisible legend just for the size explanation if we wanted, but the title handles it.
    plt.tight_layout()
    return fig

def plot_pareto_scatter(df, x_col, y_col, x_lower_better=True, y_lower_better=True, title=None):
    df = normalize_metric_columns(df)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # We need to compute the mean for each method
    df_mean = df.groupby("Method")[[x_col, y_col]].mean().dropna().reset_index()
    if df_mean.empty:
        return fig
        
    methods = df_mean["Method"].values
    xs = df_mean[x_col].values
    ys = df_mean[y_col].values
    
    # Plot points
    sns.scatterplot(x=xs, y=ys, hue=methods, s=100, ax=ax, palette="tab10", legend=False)
    
    # Find Pareto frontier
    pareto_indices = []
    for i in range(len(xs)):
        dominated = False
        for j in range(len(xs)):
            if i == j: continue
            
            x_better_or_eq = (xs[j] <= xs[i]) if x_lower_better else (xs[j] >= xs[i])
            y_better_or_eq = (ys[j] <= ys[i]) if y_lower_better else (ys[j] >= ys[i])
            
            x_strict = (xs[j] < xs[i]) if x_lower_better else (xs[j] > xs[i])
            y_strict = (ys[j] < ys[i]) if y_lower_better else (ys[j] > ys[i])
            
            if x_better_or_eq and y_better_or_eq and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
            
    # Sort pareto points by X to draw a line
    pareto_points = sorted([(xs[i], ys[i]) for i in pareto_indices], key=lambda p: p[0])
    
    if pareto_points:
        px, py = zip(*pareto_points)
        ax.plot(px, py, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Pareto Frontier")
        
    offsets = [(5, 5), (5, -10), (-5, 5), (-5, -10), (0, 10), (0, -12)]
    for i, method in enumerate(methods):
        offset = offsets[i % len(offsets)]
        ha = 'left' if offset[0] >= 0 else 'right'
        ax.annotate(method, (xs[i], ys[i]), xytext=offset, textcoords='offset points', 
                    fontsize=9, ha=ha, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0))
        
    ax.set_xlabel(f"{x_col}\n({'Lower is Better' if x_lower_better else 'Higher is Better'})")
    ax.set_ylabel(f"{y_col}\n({'Lower is Better' if y_lower_better else 'Higher is Better'})")
    if title:
        ax.set_title(title, fontweight='bold', fontsize=11)
        
    if "Runtime" in x_col:
        ax.set_xscale("log")
    
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    return fig

def plot_radar_chart(df):
    df = normalize_metric_columns(df)
    metrics_info = {
        "Attribution Runtime (sec)": False,
        "Peak Attribution Memory (MB)": False,
        "Estimated Energy Consumption (kWh)": False,
        "Deletion AUC": False,
        "Insertion AUC": True,
        "Sensitivity (Max)": False,
        "Gini Index": True
    }
    
    agg_dict = {}
    for col in metrics_info.keys():
        if col in df.columns and df[col].notna().any():
            agg_dict[col] = (col, "mean")
            
    if not agg_dict:
        fig, ax = plt.subplots(figsize=(6, 6))
        return fig
        
    df_mean = df.groupby("Method").agg(**agg_dict).reset_index()
    methods = df_mean["Method"].values
    
    # Normalize to [0, 1] where 1 is BEST
    norm_df = df_mean.copy()
    categories = []
    
    for col in agg_dict.keys():
        lower_is_better = not metrics_info[col]
        min_val = df_mean[col].min()
        max_val = df_mean[col].max()
        
        if max_val == min_val:
            norm_df[col] = 1.0
        else:
            if lower_is_better:
                # Invert: (max - val) / (max - min)
                norm_df[col] = (max_val - df_mean[col]) / (max_val - min_val)
            else:
                # Direct: (val - min) / (max - min)
                norm_df[col] = (df_mean[col] - min_val) / (max_val - min_val)
                
        label = col.replace("Attribution ", "").replace("Estimated ", "")
        categories.append(label)
        
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6.5, 6.0), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories, color='grey', size=9)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=8)
    plt.ylim(0, 1.05)
    
    # Push labels outward so they don't overlap the chart area
    ax.tick_params(axis='x', pad=15)
    
    cmap = plt.get_cmap('tab10')
    
    for i, row in norm_df.iterrows():
        values = row[list(agg_dict.keys())].values.flatten().tolist()
        values += values[:1]
        color = cmap(i % 10)
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row["Method"], color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
        
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.title("Normalized Metric Comparison\n(Outer = Better)", size=12, fontweight='bold', y=1.15)
    
    # Shrink the chart area explicitly so labels and legend fit inside the figure area
    fig.subplots_adjust(left=0.2, right=0.75, top=0.75, bottom=0.2)
    return fig

def plot_method_runtime_log(df, title="Runtime Comparison (Log Scale)"):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    
    # Sort methods by median runtime
    order = list(df.groupby("Method")[runtime_col].median().sort_values(ascending=True).index)
    
    stats = []
    for method in order:
        vals = df[df["Method"] == method][runtime_col].values
        vals = vals[vals > 0]  # Positive values only for log scale
        if len(vals) == 0:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        q1_val = float(np.percentile(vals, 25))
        q3_val = float(np.percentile(vals, 75))
        
        stats.append({
            "label": method,
            "med": mean_val,
            "q1": q1_val,
            "q3": q3_val,
            "whislo": max(min_val, mean_val - std_val),
            "whishi": min(max_val, mean_val + std_val),
            "fliers": []
        })
        
    fig, ax = plt.subplots(figsize=(10, 6))
    if stats:
        bp = ax.bxp(stats, vert=False, patch_artist=True, showmeans=False)
        
        # Style elements to match the requested look
        for patch in bp['boxes']:
            patch.set_facecolor('#8ecae6')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)
            
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.5)
            
        for line in bp['whiskers']:
            line.set_color('black')
            line.set_linestyle('--')
            
        for line in bp['caps']:
            line.set_color('black')
            
        def format_label(val):
            if val == 0: return "0.00"
            if val >= 10: return f"{val:.1f}"
            if val >= 0.1: return f"{val:.2f}"
            if val >= 0.001: return f"{val:.3f}"
            return f"{val:.4f}"
            
        for i, s in enumerate(stats):
            y = i + 1
            ax.text(s["med"], y + 0.30, format_label(s["med"]), ha='center', va='bottom', fontsize=8, color='black', alpha=0.9, fontweight='bold')
            if s["whislo"] < s["med"]:
                ax.text(s["whislo"], y - 0.33, format_label(s["whislo"]), ha='center', va='top', fontsize=7, color='black', alpha=0.75)
            if s["whishi"] > s["med"]:
                ax.text(s["whishi"], y - 0.33, format_label(s["whishi"]), ha='center', va='top', fontsize=7, color='black', alpha=0.75)

    ax.set_xscale("log")
    ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    ax.set_xlabel("Runtime (log scale, seconds)")
    ax.set_ylabel("XAI Method")
    ax.grid(True, ls="-", alpha=0.2)
    plt.tight_layout()
    return fig

def plot_method_memory(df, title="Peak Memory Overhead"):
    df = normalize_metric_columns(df)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    
    valid_df = df[df[memory_col].notna()].copy()
    if valid_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Memory profiling disabled or not available", ha='center', va='center', fontsize=12, color='gray')
        ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
        ax.axis('off')
        return fig

    # Sort methods by median memory
    order = list(valid_df.groupby("Method")[memory_col].median().sort_values(ascending=True).index)
    
    stats = []
    for method in order:
        vals = valid_df[valid_df["Method"] == method][memory_col].values
        vals = np.array([v for v in vals if v is not None and pd.notna(v) and v >= 0], dtype=float)
        if len(vals) == 0:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        q1_val = float(np.percentile(vals, 25))
        q3_val = float(np.percentile(vals, 75))
        
        stats.append({
            "label": method,
            "med": mean_val,
            "q1": q1_val,
            "q3": q3_val,
            "whislo": max(min_val, mean_val - std_val),
            "whishi": min(max_val, mean_val + std_val),
            "fliers": []
        })
        
    fig, ax = plt.subplots(figsize=(10, 6))
    if stats:
        bp = ax.bxp(stats, vert=False, patch_artist=True, showmeans=False)
        
        # Style elements
        for patch in bp['boxes']:
            patch.set_facecolor('#ffb5a7')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)
            
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.5)
            
        for line in bp['whiskers']:
            line.set_color('black')
            line.set_linestyle('--')
            
        for line in bp['caps']:
            line.set_color('black')
            
        def format_label(val):
            if val == 0: return "0.0"
            if val >= 10: return f"{val:.1f}"
            return f"{val:.2f}"
            
        # Draw labels: std below, mean above
        for i, s in enumerate(stats):
            y = i + 1
            ax.text(s["med"], y + 0.30, format_label(s["med"]), ha='center', va='bottom', fontsize=8, color='black', alpha=0.9, fontweight='bold')
            if s["whislo"] < s["med"]:
                ax.text(s["whislo"], y - 0.33, format_label(s["whislo"]), ha='center', va='top', fontsize=7, color='black', alpha=0.75)
            if s["whishi"] > s["med"]:
                ax.text(s["whishi"], y - 0.33, format_label(s["whishi"]), ha='center', va='top', fontsize=7, color='black', alpha=0.75)

    ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    ax.set_xlabel("Peak Attribution Memory (MB)")
    ax.set_ylabel("XAI Method")
    ax.grid(True, ls="-", alpha=0.2)
    plt.tight_layout()
    return fig

def plot_model_comparison_grouped(df, title="Architecture Efficiency Comparison"):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort methods (X-axis) by mean runtime (small to big)
    methods_order = list(df.groupby("Method")[runtime_col].mean().sort_values(ascending=True).index)
    # Sort models (Hue/Legend) by mean runtime (small to big)
    hue_order = list(df.groupby("Model")[runtime_col].mean().sort_values(ascending=True).index)
    
    sns.barplot(
        data=df, 
        x="Method", 
        order=methods_order,
        y=runtime_col, 
        hue="Model", 
        hue_order=hue_order,
        palette="colorblind", 
        ax=ax, 
        edgecolor="black",
        errorbar="sd"
    )
    ax.set_yscale("log")
    ax.set_ylabel("Mean Runtime (log scale, seconds)")
    ax.set_xlabel("XAI Method")
    
    # Put values on top of the bars dynamically based on height with a background mask
    bg_color = ax.get_facecolor()
    
    # Adaptive label rotation to prevent collisions
    num_hues = len(ax.containers)
    rot = 90 if num_hues > 3 else 0
    font_sz = 7 if num_hues > 4 else 8
    pad_val = 5 if rot == 90 else 3
    
    # Add top margin to prevent rotated labels clipping
    ax.margins(y=0.25 if rot == 90 else 0.15)
    
    for container in ax.containers:
        labels = []
        for rect in container:
            height = rect.get_height()
            if math.isnan(height) or height <= 0:
                labels.append("")
            elif height < 0.1:
                labels.append(f"{height:.3f}s")
            else:
                labels.append(f"{height:.2f}s")
        bar_labels = ax.bar_label(container, labels=labels, padding=pad_val, fontsize=font_sz, rotation=rot)
        for label in bar_labels:
            label.set_bbox(dict(facecolor=bg_color, edgecolor='none', pad=0.8, alpha=0.85))
            
    ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    ax.legend(title="Model Architecture", loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, ls="-", alpha=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_model_memory_comparison_grouped(df, title="Architecture Peak Memory Comparison"):
    df = normalize_metric_columns(df)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort methods (X-axis) by mean memory (small to big)
    methods_order = list(df.groupby("Method")[memory_col].mean().sort_values(ascending=True).index)
    # Sort models (Hue/Legend) by mean memory (small to big)
    hue_order = list(df.groupby("Model")[memory_col].mean().sort_values(ascending=True).index)
    
    sns.barplot(
        data=df, 
        x="Method", 
        order=methods_order,
        y=memory_col, 
        hue="Model", 
        hue_order=hue_order,
        palette="colorblind", 
        ax=ax, 
        edgecolor="black",
        errorbar="sd"
    )
    ax.set_ylabel("Mean Peak Memory (MB)")
    ax.set_xlabel("XAI Method")
    
    # Put values on top of the bars dynamically based on height with a background mask
    bg_color = ax.get_facecolor()
    
    # Adaptive label rotation to prevent collisions
    num_hues = len(ax.containers)
    rot = 90 if num_hues > 3 else 0
    font_sz = 7 if num_hues > 4 else 8
    pad_val = 5 if rot == 90 else 3
    
    # Add top margin to prevent rotated labels clipping
    ax.margins(y=0.25 if rot == 90 else 0.15)
    
    for container in ax.containers:
        labels = []
        for rect in container:
            height = rect.get_height()
            if math.isnan(height) or height <= 0:
                labels.append("")
            else:
                labels.append(f"{height:.1f}MB")
        bar_labels = ax.bar_label(container, labels=labels, padding=pad_val, fontsize=font_sz, rotation=rot)
        for label in bar_labels:
            label.set_bbox(dict(facecolor=bg_color, edgecolor='none', pad=0.8, alpha=0.85))
            
    ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    ax.legend(title="Model Architecture", loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, ls="-", alpha=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_image_size_runtime_scaling(summary_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_df = summary_df.sort_values("Resolution")
    ax.errorbar(
        summary_df["Resolution"],
        summary_df["Mean Attribution Runtime (sec)"],
        yerr=summary_df["Attribution Runtime Std (sec)"],
        marker="o",
        linestyle="-",
        linewidth=2,
        capsize=4,
        color="#1f77b4"
    )
    ax.set_xticks(summary_df["Resolution"])
    ax.set_xticklabels([str(r) for r in summary_df["Resolution"]])
    
    ax.set_xlabel("Resolution (px)")
    ax.set_ylabel("Mean Attribution Runtime (sec)")
    ax.set_title("Overall Runtime Scaling by Resolution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

def plot_image_size_memory_scaling(summary_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_df = summary_df.sort_values("Resolution")
    ax.errorbar(
        summary_df["Resolution"],
        summary_df["Mean Peak Attribution Memory (MB)"],
        yerr=summary_df["Peak Memory Std (MB)"],
        marker="o",
        linestyle="-",
        linewidth=2,
        capsize=4,
        color="#d62728"
    )
    ax.set_xticks(summary_df["Resolution"])
    ax.set_xticklabels([str(r) for r in summary_df["Resolution"]])
    
    ax.set_xlabel("Resolution (px)")
    ax.set_ylabel("Mean Peak Attribution Memory (MB)")
    ax.set_title("Overall Memory Scaling by Resolution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

def plot_runtime_distribution(df):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Method", y=runtime_col, ax=ax, color="#8ecae6")
    ax.set_title("Attribution Runtime Distribution", fontsize=14, fontweight='bold')
    ax.set_ylabel("Attribution Runtime (sec)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_runtime_memory_scatter(df, figsize=(9, 6)):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    
    # Group by configuration (Method, Model, Resolution) and average the metrics (across all images)
    group_cols = ["Method", "Model", "Resolution"]
    df_config = df.groupby(group_cols).agg({runtime_col: "mean", memory_col: "mean"}).reset_index()
    
    # Combine Model and Resolution into a single column for marker style grouping
    df_config["Model (Resolution)"] = df_config["Model"] + " (" + df_config["Resolution"] + ")"
    
    fig, ax = plt.subplots(figsize=figsize)
    # Style represents Model (Resolution), Hue represents Method. Constant size (s=110)
    sns.scatterplot(
        data=df_config, 
        x=runtime_col, 
        y=memory_col, 
        hue="Method", 
        style="Model (Resolution)", 
        s=110, 
        ax=ax, 
        edgecolor="black"
    )
    ax.set_title("Runtime vs Peak Attribution Memory", fontsize=14, fontweight='bold')
    ax.set_xlabel("Mean Attribution Runtime (sec)")
    ax.set_ylabel("Mean Peak Attribution Memory (MB)")  # Use logarithmic scales to prevent extreme outliers from squashing the plot
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    # Extract legend handles and insert an empty spacer row between different categories
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    new_labels = []
    
    for h, l in zip(handles, labels):
        # Prepend a blank row when moving to the Model (Resolution) section
        if l in ["Model (Resolution)", "Model", "Input Size (px)"] and len(new_labels) > 0:
            new_handles.append(Patch(color='none', label=''))
            new_labels.append('')
        new_handles.append(h)
        new_labels.append(l)
        
    ax.legend(new_handles, new_labels, loc='upper left', bbox_to_anchor=(1, 1), labelspacing=0.65)
    ax.grid(True, ls="-", alpha=0.15)
    plt.tight_layout()
    return fig

def plot_combined_batches_comparison(df):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    df["Model_Method"] = df["Model"] + "\n(" + df["Method"] + ")"
    df_sorted = df.sort_values(by=["Model_Method", "Batch"])
    
    sns.barplot(
        data=df_sorted, 
        x="Model_Method", 
        y=runtime_col, 
        hue="Batch", 
        palette="viridis", 
        ax=ax, 
        edgecolor="black"
    )
    ax.set_title("Attribution Runtime Comparison Across Batches", fontsize=14, fontweight='bold', family='serif')
    ax.set_xlabel("Model & Method")
    ax.set_ylabel("Attribution Runtime (sec)")
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, ls="-", alpha=0.15)
    plt.tight_layout()
    return fig

def plot_combined_batches_memory(df):
    df = normalize_metric_columns(df)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    
    valid_df = df[df[memory_col].notna()].copy()
    if valid_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Memory profiling disabled or not available", ha='center', va='center', fontsize=12, color='gray')
        ax.set_title("Peak Attribution Memory Comparison Across Batches", fontsize=14, fontweight='bold', family='serif')
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(12, 7))
    valid_df["Model_Method"] = valid_df["Model"] + "\n(" + valid_df["Method"] + ")"
    df_sorted = valid_df.sort_values(by=["Model_Method", "Batch"])
    
    sns.barplot(
        data=df_sorted, 
        x="Model_Method", 
        y=memory_col, 
        hue="Batch", 
        palette="viridis", 
        ax=ax, 
        edgecolor="black"
    )
    ax.set_title("Peak Attribution Memory Comparison Across Batches", fontsize=14, fontweight='bold', family='serif')
    ax.set_xlabel("Model & Method")
    ax.set_ylabel("Peak Memory (MB)")
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, ls="-", alpha=0.15)
    plt.tight_layout()
    return fig
