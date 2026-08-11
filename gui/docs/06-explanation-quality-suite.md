# 06 - Explanation Quality & Faithfulness Suite

## 1. Overview & Purpose

Evaluating Explainable AI methods requires balancing computational cost against explanation fidelity. A method that executes in 2 milliseconds is of limited value if its attribution maps are uninformative, noisy, or unfaithful to the model's actual decision process.

To provide a comprehensive trade-off analysis, the framework includes a post-hoc explanation evaluation suite implemented in [`gui/backend/quality_runner.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/quality_runner.py). These metrics run **strictly off the timing clock** to keep runtime and memory benchmarks clean.

---

## 2. Mathematical Formulations & Evaluation Metrics

```
┌───────────────────────────┬────────────────────────────┬─────────────────────────────┐
│ Metric                    │ Measurement Focus          │ Optimal Value Behavior      │
├───────────────────────────┼────────────────────────────┼─────────────────────────────┤
│ Gini Index (Sparsity)     │ Spatial concentration      │ Closer to 1.0 (Higher)      │
│ Deletion AUC              │ Faithfulness to salient px │ Closer to 0.0 (Lower)       │
│ Insertion AUC             │ Information sufficiency    │ Closer to 1.0 (Higher)      │
│ Sensitivity (Max)         │ Attribution stability      │ Closer to 0.0 (Lower)       │
│ Infidelity                │ Logit perturbation MSE     │ Closer to 0.0 (Lower)       │
└───────────────────────────┴────────────────────────────┴─────────────────────────────┘
```

### 2.1 Gini Index (Attribution Sparsity)
The Gini index evaluates the spatial focus and sharpness of the attribution mask. High sparsity indicates that the explanation highlights a concise, localized set of features rather than distributing diffuse weight across the entire image.

Given flattened, sorted absolute attribution values $|a|_{(1)} \le |a|_{(2)} \le \dots \le |a|_{(N)}$ for $N$ pixels:

$$\text{Gini} = \frac{2 \sum_{i=1}^{N} i \cdot |a|_{(i)}}{N \sum_{i=1}^{N} |a|_{(i)}} - \frac{N + 1}{N}$$

- **Range**: $[0, 1]$.
- **Interpretation**: A score near `1.0` represents sharp, compact spatial focus; a score near `0.0` indicates uniform blur.

---

### 2.2 Deletion AUC (Faithfulness Curve)
Deletion AUC measures how rapidly model prediction confidence degrades as the most important pixels (ranked by attribution magnitude) are progressively removed and replaced by a baseline (e.g., zeros/black).

1. Sort all pixel indices $i \in \{1, \dots, N\}$ in descending order of attribution magnitude $|a_i|$.
2. In $K$ discrete steps (default: 10 steps), mask the top $k\%$ most salient pixels.
3. Compute the model's Softmax probability for the target class at each step: $p_k = P(y = \text{target} \mid \tilde{x}_k)$.
4. Compute the Area Under the Curve (AUC) using the trapezoidal rule over the degradation trajectory:

$$\text{Deletion AUC} = \int_{0}^{1} P(y = \text{target} \mid \tilde{x}_\alpha) \, d\alpha$$

- **Interpretation**: **Lower is better**. A rapid drop in prediction confidence proves that the identified pixels were genuinely critical to the model's decision.

---

### 2.3 Insertion AUC (Sufficiency Curve)
Insertion AUC measures how rapidly model prediction confidence is restored as the most important pixels are progressively inserted into a blank baseline image.

1. Start from a blank reference baseline image $x_0$ (e.g., all zeros).
2. In $K$ discrete steps, restore the top $k\%$ most salient pixels from the original input $x$.
3. Measure target class confidence recovery $p_k = P(y = \text{target} \mid \hat{x}_k)$ at each step.
4. Calculate the Area Under the Curve:

$$\text{Insertion AUC} = \int_{0}^{1} P(y = \text{target} \mid \hat{x}_\alpha) \, d\alpha$$

- **Interpretation**: **Higher is better**. A rapid rise in confidence indicates that the top-ranked pixels alone contain sufficient information to trigger the classification.

---

### 2.4 Sensitivity (Max)
Sensitivity measures the worst-case instability of the explanation when the input image is subjected to subtle, imperceptible perturbations (e.g., small radius Gaussian noise or radius shifts):

$$\text{Sensitivity}(x) = \max_{||\delta|| \le \epsilon} \frac{||A(x + \delta) - A(x)||_F}{||\delta||_F}$$

The evaluation leverages Captum's `captum.metrics.sensitivity_max` utility.
- **Interpretation**: **Lower is better**. Lower sensitivity indicates that the explanation is robust against input noise and adversarial jitter.

---

### 2.5 Infidelity
Infidelity measures the Mean Squared Error (MSE) between the difference in model logit predictions under random Gaussian perturbations and the dot product of the perturbation vector with the attribution map:

$$\text{Infidelity} = \mathbb{E}_{\delta} \left[ \left( \delta^T A(x) - \left( f(x) - f(x - \delta) \right) \right)^2 \right]$$

Implemented via `captum.metrics.infidelity`.
- **Interpretation**: **Lower is better**. Captures how linearly consistent the attribution map is with local model output changes.
- **Incompatibility Warning on Region-Based Methods**: Infidelity evaluates fine-grained Gaussian perturbations ($\delta^T A(x)$). For region-based or superpixel explanation methods (defined in `config.region_based_methods = ["Occlusion", "LIME"]`), this formulation is mathematically ill-suited: coarse patch-level weights cannot track high-frequency pixel noise, producing distorted, invalid scores that cannot be meaningfully compared against pixel-level methods.

---

## 3. Key Files & Directory Mapping

- [`gui/backend/quality_runner.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/quality_runner.py) — Complete post-hoc quality metrics implementation (Gini, AUC, Sensitivity, Infidelity).
- [`gui/backend/benchmark_runner.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/benchmark_runner.py#L31) — Orchestrates the execution sequence, invoking quality calculations off-the-clock.

---

## 4. Related Documentation

- [`05-benchmark-execution-engine.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/05-benchmark-execution-engine.md) — Details the execution loop that feeds attributions into the quality suite.
- [`07-analysis-and-pareto.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/07-analysis-and-pareto.md) — Explains how quality metrics are plotted against attribution runtimes on Pareto trade-off frontiers.
- [`12-in-app-documentation-and-taxonomy.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/12-in-app-documentation-and-taxonomy.md) — In-app documentation view summarizing quality formulas for end users.
