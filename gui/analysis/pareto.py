"""
Pareto Analysis Module
Contains mathematical logic for computing Pareto optimal frontiers and ranking XAI methods based on runtime vs quality trade-offs.
"""
import pandas as pd

from gui.data.processing import normalize_metric_columns

def compute_pareto_ranking(df, runtime_col):
    df = normalize_metric_columns(df)
    metrics = [
        ("Deletion AUC", True),
        ("Insertion AUC", False),
        ("Sensitivity (Max)", True),
        ("Gini Index", False)
    ]
    
    if runtime_col not in df.columns:
        return pd.DataFrame()
        
    all_methods = df["Method"].unique()
    ranking_data = []
    
    # Pre-compute valid df_means for each metric to match plot_pareto_scatter exactly
    metric_dfs = {}
    for metric_col, y_lower_better in metrics:
        if metric_col in df.columns:
            metric_dfs[metric_col] = df.groupby("Method")[[runtime_col, metric_col]].mean().dropna().reset_index()
            
    for method in all_methods:
        row = {"Method": method}
        total_fronts = 0
        evaluated_metrics = 0
        
        for metric_col, y_lower_better in metrics:
            if metric_col not in metric_dfs:
                continue
                
            df_mean = metric_dfs[metric_col]
            if method not in df_mean["Method"].values:
                # If this method doesn't have valid data for this metric, mark N/A
                col_name = f"Runtime–{metric_col.replace(' (Max)', '')}"
                row[col_name] = "N/A"
                continue
                
            evaluated_metrics += 1
            
            methods = df_mean["Method"].values
            xs = df_mean[runtime_col].values
            ys = df_mean[metric_col].values
            
            i = list(methods).index(method)
            
            dominated = False
            for j in range(len(xs)):
                if i == j: continue
                
                # We always consider runtime lower is better
                x_better_or_eq = (xs[j] <= xs[i])
                y_better_or_eq = (ys[j] <= ys[i]) if y_lower_better else (ys[j] >= ys[i])
                
                x_strict = (xs[j] < xs[i])
                y_strict = (ys[j] < ys[i]) if y_lower_better else (ys[j] > ys[i])
                
                if x_better_or_eq and y_better_or_eq and (x_strict or y_strict):
                    dominated = True
                    break
                    
            is_optimal = not dominated
            col_name = f"Runtime–{metric_col.replace(' (Max)', '')}"
            row[col_name] = "Pareto-optimal" if is_optimal else "Dominated"
            if is_optimal:
                total_fronts += 1
                
        if len(row) > 1:
            row["Overall"] = f"{total_fronts}/{evaluated_metrics}"
            ranking_data.append(row)
            
    if not ranking_data:
        return pd.DataFrame()
        
    ranking_df = pd.DataFrame(ranking_data)
    
    # Sort so methods with the most pareto optimal fronts are at the top
    def _parse_fraction(val):
        if val == "N/A": return -1.0
        parts = val.split('/')
        if len(parts) == 2 and parts[1] != '0':
            return float(parts[0]) / float(parts[1])
        return 0.0
        
    if "Overall" in ranking_df.columns:
        ranking_df["_sort_val"] = ranking_df["Overall"].apply(_parse_fraction)
        ranking_df = ranking_df.sort_values(by="_sort_val", ascending=False).drop(columns=["_sort_val"])
        
    return ranking_df
