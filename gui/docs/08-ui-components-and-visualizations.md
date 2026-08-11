# 08 - UI Components, Charts & Visualizations

## 1. Overview & Purpose

The dashboard user interface combines custom glassmorphism HTML/CSS components with statistical **Matplotlib** and **Seaborn** visualizations. 

This module describes the modular UI component system in [`gui/components/`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components), detailing card renderers, live JavaScript timers, statistical chart generation, and data table formatters.

---

## 2. Component System Architecture

```
gui/components/
├── cards.py   # HTML/CSS glassmorphism summary cards, live elapsed timer, and per-image result collages
├── plots.py   # High-resolution Matplotlib/Seaborn statistical plots and Pareto trade-off figures
├── tables.py  # Interactive Streamlit data tables, column formatters, and style mappers
└── media.py   # Base64 image loaders and thumbnail encoders
```

---

## 3. Visual Components & Card Renderers (`cards.py`)

[`gui/components/cards.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components/cards.py) encapsulates HTML template generation to maintain clean separation between presentation and page logic:

- **Live Elapsed Timer** (`render_live_elapsed_timer`): An asynchronous JavaScript DOM timer embedded via `st.components.v1.html` that counts execution elapsed time smoothly without triggering Streamlit reruns.
- **Environment Summary Card** (`render_environment_summary`): Displays host OS, Python, PyTorch, CUDA runtime version, active compute device, CPU name, GPU models, and TDP power ratings in a styled grid.
- **Configuration Summary Card** (`render_configuration_summary`): Outlines active batch parameters (image counts, model backbones, XAI methods, resolution lists, quality metric toggles, warmup counts, repeat counts, and task queuing strategy).
- **Parameters Mapping Card** (`render_parameters_mapping_table`): For parameterized multi-version runs, presents an explicit parameter breakdown mapping each unique method instance to its configured hyperparameters (steps, batch sizes, patch sizes, segments).
- **Per-Image Result Groups** (`render_result_group`): Renders side-by-side heatmaps alongside granular metrics tables for every evaluated image.

---

## 4. Statistical Plotting Suite (`plots.py`)

All charts in [`gui/components/plots.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components/plots.py) adhere to a dark-mode palette matching the application theme (Background: `#0b0f19`, Grid lines: `#1e293b`, Primary accent: `#0d9488`, Secondary accent: `#38bdf8`, Text: `#e2e8f0`):

| Plot Function | Visualization Type | Analytical Purpose |
|:---|:---|:---|
| `plot_model_comparison_grouped` | Grouped Bar Chart | Compares mean attribution runtime across model backbones and methods. |
| `plot_model_memory_comparison_grouped`| Grouped Bar Chart | Compares peak allocation memory (MB) across architectures. |
| `plot_runtime_memory_scatter` | 2D Scatter Plot | Maps execution speed against peak memory footprint. |
| `plot_pareto_scatter` | Frontier Step Scatter | Plots speed vs quality (e.g. Deletion AUC) with non-dominated Pareto frontier steps. |
| `plot_bubble_chart` | 3D Bubble Chart | Encodes Runtime (X), Quality/Memory (Y), and Model Parameters (Bubble Radius). |
| `plot_image_size_runtime_scaling` | Line / Error-Bar Plot| Tracks attribution runtime scaling across 25 image resolutions (32px → 1024px). |
| `plot_image_size_memory_scaling` | Line / Error-Bar Plot| Tracks peak VRAM/RAM scaling as input dimensions grow. |
| `plot_method_runtime_log` | Log-Scale Bar Chart | Visualizes runtime variance across orders of magnitude for fast vs slow methods. |

---

## 5. Key Files & Directory Mapping

- [`gui/components/cards.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components/cards.py) — Glassmorphism HTML summary cards, active run telemetry, and per-image result collages.
- [`gui/components/plots.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components/plots.py) — Matplotlib and Seaborn statistical plotting engine.
- [`gui/components/tables.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components/tables.py) — Interactive Streamlit data tables and column precision formatters.
- [`gui/components/media.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components/media.py) — Base64 image encoding and thumbnail formatting helpers.
- [`gui/assets/styles.css`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/styles.css) — Global CSS styling and dark theme variables.

---

## 6. Related Documentation

- [`01-architecture-and-lifecycle.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/01-architecture-and-lifecycle.md) — Explains the application layout and CSS injection mechanisms.
- [`07-analysis-and-pareto.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/07-analysis-and-pareto.md) — Details the mathematical Pareto frontier equations visualized by `plots.py`.
- [`10-export-systems.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/10-export-systems.md) — Documents how these Matplotlib figures are converted to vector flowables for PDF reports.
