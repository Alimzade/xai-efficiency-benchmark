"""
Explanation Quality & Fidelity Metrics Module
Calculates post-hoc quality metrics (e.g. Gini Index / Sparsity) on generated attribution maps.
All functions run strictly in post-processing outside the timing and memory benchmarking loops.
"""

import numpy as np
import torch


def compute_gini_index(attribution):
    """
    Computes the Gini Coefficient (Sparsity) of an attribution map.
    
    A Gini index close to 1.0 indicates a highly concentrated/sparse heatmap (high focus).
    A Gini index close to 0.0 indicates a uniform/dispersed heatmap (low focus).
    
    Args:
        attribution: PyTorch Tensor or NumPy array of attribution values.
        
    Returns:
        float: Rounded Gini coefficient in range [0.0, 1.0], or None if input is invalid.
    """
    if attribution is None:
        return None
        
    if hasattr(attribution, "detach"):
        attribution = attribution.detach().cpu().numpy()
        
    # Take absolute values of attributions and flatten to 1D
    arr = np.abs(np.asarray(attribution, dtype=np.float64)).flatten()
    n = arr.size
    
    if n == 0:
        return None
    
    sum_arr = np.sum(arr)
    if sum_arr == 0:
        return 0.0
        
    # Sort attribution values in ascending order
    arr_sorted = np.sort(arr)
    index = np.arange(1, n + 1, dtype=np.float64)
    
    # Formula: sum((2*i - n - 1) * x_i) / (n * sum(x_i))
    gini = np.sum((2.0 * index - n - 1.0) * arr_sorted) / (n * sum_arr)
    return float(round(gini, 4))


def compute_quality_metrics(attribution, model=None, input_tensor=None, target_class=None, selected_metrics=None):
    """
    Dispatcher for computing requested explanation quality metrics on a generated attribution.
    
    Args:
        attribution: Generated attribution map tensor/array.
        model: PyTorch model instance (optional, needed for forward-pass metrics).
        input_tensor: PyTorch input image tensor (optional).
        target_class: Target prediction class index (optional).
        selected_metrics (list): List of metric names to compute (e.g. ["Gini Index (Sparsity)"]).
        
    Returns:
        dict: Dictionary of calculated quality metric scores.
    """
    results = {}
    if not selected_metrics:
        return results
        
    for metric_name in selected_metrics:
        norm_name = metric_name.strip().lower()
        if "gini" in norm_name or "sparsity" in norm_name:
            results["Gini Index"] = compute_gini_index(attribution)
            
    return results
