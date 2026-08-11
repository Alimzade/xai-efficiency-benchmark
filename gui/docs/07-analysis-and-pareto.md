# 07 - Data Analysis, Metrics Aggregation & Pareto Optimization

## 1. Overview & Purpose

Individual benchmark tasks produce granular runtime, memory, and quality data points. To extract actionable research insights across hundreds of permutations, the analysis subsystem aggregates statistical distributions and identifies **Pareto-optimal** explanation methods.

This module explains how raw benchmark records are aggregated in [`gui/analysis/metrics.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/analysis/metrics.py) and how multi-objective trade-off frontiers are computed in [`gui/analysis/pareto.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/analysis/pareto.py).

---

## 2. Statistical Aggregation Pipeline

Benchmark runs frequently evaluate multiple images, resolutions, and architectures. The aggregator in [`gui/analysis/metrics.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/analysis/metrics.py) organizes records into high-level analytical representations:

1. **Configuration Normalization**: Maps legacy column variants into canonical identifiers (`Attribution Runtime (sec)`, `Peak Attribution Memory (MB)`).
2. **Grouping & Reduction**: Groups observations by `(Model, Method, Input Size (px))` to compute:
   - **Central Tendency**: Mean and Median attribution runtime.
   - **Variance & Stability**: Standard Deviation across measured repetitions.
   - **Resource Footprint**: Mean peak memory allocations and mean estimated energy (kWh).
   - **Quality Mean**: Averages for Deletion AUC, Insertion AUC, Gini Index, Sensitivity, and Infidelity.

---

## 3. Pareto Optimality & Trade-Off Frontier Analysis

When choosing an XAI method for production or resource-constrained edge devices, practitioners face conflicting objectives: maximizing explanation quality while minimizing execution latency and memory overhead.

### 3.1 Mathematical Definition of Pareto Dominance
An attribution configuration $A$ is said to **strictly dominate** configuration $B$ ($A \succ B$) if and only if:
1. $A$ is no worse than $B$ across all evaluated objectives:
   $$\forall i \in \{1, \dots, M\}, \quad \text{score}_i(A) \le \text{score}_i(B) \quad (\text{assuming minimization})$$
2. $A$ is strictly better than $B$ on at least one objective:
   $$\exists j \in \{1, \dots, M\}, \quad \text{score}_j(A) < \text{score}_j(B)$$

A configuration is **Pareto-optimal (non-dominated)** if no other configuration exists that improves one metric without worsening another. The set of all non-dominated points constitutes the **Pareto Frontier**.

```
    Quality (e.g., Lower Deletion AUC is Better)
        ▲
        │  * (Suboptimal)
        │
   Best │  ★ Method A (Pareto Frontier)
        │    \
        │     \
        │      ★ Method B (Pareto Frontier)
        │        \
        │         \
  Worst │          ★ Method C (Ultra Fast)
        └───────────────────────────────────► Runtime (sec)
          Fastest                     Slowest
```

### 3.2 Implemented Trade-off Axes
[`gui/analysis/pareto.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/analysis/pareto.py) computes 2D Pareto frontiers across multiple critical trade-off dimensions:

- **Speed vs. Faithfulness**: `Attribution Runtime` vs `Deletion AUC` (Minimizing both).
- **Speed vs. Sufficiency**: `Attribution Runtime` (min) vs `Insertion AUC` (max).
- **Speed vs. Robustness**: `Attribution Runtime` vs `Sensitivity (Max)` (Minimizing both).
- **Speed vs. Memory**: `Attribution Runtime` vs `Peak Attribution Memory` (Minimizing both).

Points along the calculated frontier are tagged with boolean indicators (`is_pareto_optimal = True`) and visually emphasized in scatter plots with custom highlighted markers and connected step-lines.

---

## 4. Key Files & Directory Mapping

- [`gui/analysis/metrics.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/analysis/metrics.py) — Multi-run grouping, central tendency aggregation, and statistical summarization.
- [`gui/analysis/pareto.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/analysis/pareto.py) — Vectorized Pareto frontier extraction algorithms (`compute_pareto_frontier`, `is_pareto_efficient`).
- [`gui/components/plots.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components/plots.py) — Visual rendering of Pareto scatter plots, bubble charts, and scaling curves.

---

## 5. Related Documentation

- [`05-benchmark-execution-engine.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/05-benchmark-execution-engine.md) — Generates the raw runtime and memory records analyzed here.
- [`06-explanation-quality-suite.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/06-explanation-quality-suite.md) — Details the mathematical definitions of the quality metrics used as Pareto axes.
- [`08-ui-components-and-visualizations.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/08-ui-components-and-visualizations.md) — Documents the UI components and charts that render Pareto frontiers.
