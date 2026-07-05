"""
Explanation Quality & Fidelity Metrics Module
Calculates post-hoc quality metrics (Gini, Deletion AUC, Insertion AUC, Infidelity)
outside the timed XAI benchmarking clock.
"""

import logging
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from captum.metrics import infidelity

logger = logging.getLogger(__name__)


@dataclass
class QualityContext:
    attribution: torch.Tensor
    model: torch.nn.Module
    input_tensor: torch.Tensor
    target_class: int
    method_key: str
    device: torch.device
    
    # Lazy-cached artifacts
    _spatial_importance: Optional[torch.Tensor] = None
    _sorted_indices: Optional[torch.Tensor] = None
    _orig_logits: Optional[torch.Tensor] = None
    _orig_prob: Optional[float] = None

    def get_sorted_pixel_indices(self) -> torch.Tensor:
        """Computes spatial importance ranking once and caches result."""
        if self._sorted_indices is None:
            attr = self.attribution.squeeze(0) if self.attribution.dim() == 4 else self.attribution
            # Spatial sum across channels: (C, H, W) -> (H, W)
            spatial_map = torch.abs(attr).sum(dim=0)
            self._spatial_importance = spatial_map
            self._sorted_indices = torch.argsort(spatial_map.flatten(), descending=True)
        return self._sorted_indices

    def get_initial_prediction(self) -> Tuple[torch.Tensor, float]:
        """Executes clean forward pass once and caches logits and target probability."""
        if self._orig_logits is None or self._orig_prob is None:
            with torch.no_grad():
                self._orig_logits = self.model(self.input_tensor)
                probs = F.softmax(self._orig_logits, dim=1)
                self._orig_prob = float(probs[0, self.target_class].item())
        return self._orig_logits, self._orig_prob


# --- 1. SPARSITY (GINI INDEX) ---
def compute_gini_index(attribution: torch.Tensor) -> Optional[float]:
    """Computes full-precision Gini Coefficient."""
    if attribution is None:
        return None
    if hasattr(attribution, "detach"):
        attribution = attribution.detach().cpu().numpy()
        
    arr = np.abs(np.asarray(attribution, dtype=np.float64)).flatten()
    n = arr.size
    if n == 0 or np.sum(arr) == 0:
        return 0.0
        
    arr_sorted = np.sort(arr)
    index = np.arange(1, n + 1, dtype=np.float64)
    gini = np.sum((2.0 * index - n - 1.0) * arr_sorted) / (n * np.sum(arr))
    return float(gini)  # Raw unrounded float


# --- 2 & 3. DELETION AND INSERTION AUC ---
def compute_deletion_insertion_auc(
    ctx: QualityContext, 
    steps: int = 10, 
    baseline_type: str = "zero"
) -> Dict[str, Optional[float]]:
    """
    Computes Deletion and Insertion AUC using cumulative masking and cached predictions.
    """
    try:
        model = ctx.model
        input_tensor = ctx.input_tensor
        target_class = ctx.target_class
        sorted_indices = ctx.get_sorted_pixel_indices()
        _, orig_prob = ctx.get_initial_prediction()

        _, C, H, W = input_tensor.shape
        total_pixels = H * W
        step_size = max(1, total_pixels // steps)

        # Baseline generation
        if baseline_type == "zero":
            baseline = torch.zeros_like(input_tensor)
        elif baseline_type == "mean":
            baseline = torch.mean(input_tensor, dim=(-2, -1), keepdim=True).expand_as(input_tensor)
        else:
            baseline = torch.zeros_like(input_tensor)

        with torch.no_grad():
            base_prob = float(F.softmax(model(baseline), dim=1)[0, target_class].item())

        del_probs = [orig_prob]
        ins_probs = [base_prob]

        # Cumulative image tensors for in-place updates
        del_img = input_tensor.clone()
        ins_img = baseline.clone()
        input_flat = input_tensor.view(1, C, -1)
        base_flat = baseline.view(1, C, -1)

        with torch.no_grad():
            for step in range(1, steps):
                start_idx = (step - 1) * step_size
                end_idx = min(step * step_size, total_pixels)
                chunk_indices = sorted_indices[start_idx:end_idx]

                # Cumulative update: update only the newly masked chunk of pixels
                del_img.view(1, C, -1)[:, :, chunk_indices] = base_flat[:, :, chunk_indices]
                ins_img.view(1, C, -1)[:, :, chunk_indices] = input_flat[:, :, chunk_indices]

                del_probs.append(float(F.softmax(model(del_img), dim=1)[0, target_class].item()))
                ins_probs.append(float(F.softmax(model(ins_img), dim=1)[0, target_class].item()))

        del_probs.append(base_prob)
        ins_probs.append(orig_prob)

        # Normalize probability curves by initial model confidence (orig_prob)
        # Ensures AUC ranges [0.0, 1.0] and remains invariant to raw ImageNet top-1 probability magnitude
        if orig_prob > 1e-6:
            del_probs_norm = [min(1.0, max(0.0, p / orig_prob)) for p in del_probs]
            ins_probs_norm = [min(1.0, max(0.0, p / orig_prob)) for p in ins_probs]
        else:
            del_probs_norm = del_probs
            ins_probs_norm = ins_probs

        x_steps = np.linspace(0, 1, len(del_probs_norm))
        del_auc = float(np.trapezoid(del_probs_norm, x_steps))
        ins_auc = float(np.trapezoid(ins_probs_norm, x_steps))

        return {"Deletion AUC": del_auc, "Insertion AUC": ins_auc}

    except Exception as e:
        logger.warning(f"Error computing Deletion/Insertion AUC: {e}\n{traceback.format_exc()}")
        return {"Deletion AUC": None, "Insertion AUC": None}


# --- 4. INFIDELITY (ROBUST & SCALE-INVARIANT) ---
def compute_infidelity_score(ctx: QualityContext, n_samples: int = 10, noise_sigma: float = 0.1) -> Optional[float]:
    """Computes scale-invariant, L2-normalized Infidelity metric with Gaussian noise perturbation."""
    try:
        model = ctx.model
        input_tensor = ctx.input_tensor
        attribution = ctx.attribution
        target = ctx.target_class

        if attribution is None:
            return None

        # Clean NaN/Inf in attribution
        attribution = torch.nan_to_num(attribution, nan=0.0, posinf=0.0, neginf=0.0)

        # Resize coarse heatmaps (e.g. Grad-CAM) to input resolution
        if attribution.shape[-2:] != input_tensor.shape[-2:]:
            attr_resized = F.interpolate(attribution, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        else:
            attr_resized = attribution.clone()

        # Collapse multi-channel attribution to single spatial importance map: (1, C, H, W) -> (1, 1, H, W)
        if attr_resized.dim() == 4 and attr_resized.size(1) > 1:
            attr_spatial = torch.abs(attr_resized).mean(dim=1, keepdim=True)
        elif attr_resized.dim() == 3:
            attr_spatial = torch.abs(attr_resized).unsqueeze(0).mean(dim=1, keepdim=True)
        else:
            attr_spatial = torch.abs(attr_resized)

        # L2 Unit Norm normalization across spatial dimensions
        flat_attr = attr_spatial.view(attr_spatial.size(0), -1)
        l2_norm = torch.norm(flat_attr, p=2, dim=-1, keepdim=True).view(attr_spatial.size(0), 1, 1, 1)
        if torch.any(l2_norm == 0):
            return 0.0
        attr_normalized = attr_spatial / (l2_norm + 1e-8)

        # Generate n_samples noisy inputs for perturbation expectation
        inputs_rep = input_tensor.repeat(n_samples, 1, 1, 1)
        noise = torch.randn_like(inputs_rep) * noise_sigma
        perturbed_inputs = inputs_rep + noise

        # Mean spatial noise matching single-channel attribution map
        noise_spatial = noise.mean(dim=1, keepdim=True)

        # 1. Compute attribution dot products: (n_samples,)
        dot_products = (noise_spatial * attr_normalized).view(n_samples, -1).sum(dim=-1)

        # 2. Compute target class probability drop under perturbation: (n_samples,)
        with torch.no_grad():
            orig_prob = F.softmax(model(input_tensor), dim=1)[0, target].item()
            pert_probs = F.softmax(model(perturbed_inputs), dim=1)[:, target]
            prob_diffs = orig_prob - pert_probs

        # 3. Compute Mean Squared Error (MSE) infidelity score
        infid_score = torch.mean((dot_products - prob_diffs) ** 2).item()
        return float(infid_score)
    except Exception as e:
        logger.warning(f"Error computing Infidelity: {e}\n{traceback.format_exc()}")
        return None


# --- MAIN DISPATCHER ---
def compute_quality_metrics(
    attribution, 
    model=None, 
    input_tensor=None, 
    target_class=None, 
    method_key="", 
    device=None, 
    selected_metrics=None,
    metric_kwargs=None
) -> Dict[str, Optional[float]]:
    """Dispatcher for computing requested explanation quality metrics with execution overhead timing."""
    results = {}
    if not selected_metrics:
        return results

    kwargs = metric_kwargs or {}
    start_time = time.perf_counter()

    ctx = QualityContext(
        attribution=attribution,
        model=model,
        input_tensor=input_tensor,
        target_class=target_class,
        method_key=method_key,
        device=device
    )

    norm_metrics = [m.strip().lower() for m in selected_metrics]

    # 1. Sparsity / Gini
    if any("gini" in m or "sparsity" in m for m in norm_metrics):
        results["Gini Index"] = compute_gini_index(attribution)

    # 2 & 3. Deletion and Insertion AUC
    if (any("deletion" in m for m in norm_metrics) or any("insertion" in m for m in norm_metrics)) \
            and model is not None and input_tensor is not None:
        del_ins_res = compute_deletion_insertion_auc(
            ctx, 
            steps=kwargs.get("deletion_steps", 10), 
            baseline_type=kwargs.get("baseline_type", "zero")
        )
        if any("deletion" in m for m in norm_metrics):
            results["Deletion AUC"] = del_ins_res.get("Deletion AUC")
        if any("insertion" in m for m in norm_metrics):
            results["Insertion AUC"] = del_ins_res.get("Insertion AUC")

    # 4. Infidelity
    if any("infidelity" in m for m in norm_metrics) and model is not None and input_tensor is not None:
        results["Infidelity"] = compute_infidelity_score(
            ctx, 
            n_samples=kwargs.get("n_samples", 10),
            noise_sigma=kwargs.get("noise_sigma", 0.1)
        )

    # Record evaluation overhead runtime (sec)
    eval_overhead = time.perf_counter() - start_time
    results["Quality Eval Time (sec)"] = eval_overhead

    return results

