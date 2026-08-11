# 04 - Benchmark Configuration, Hyperparameters & Task Queuing

## 1. Overview & Purpose

The configuration view is the experimental staging ground of the benchmark. It allows researchers to specify test images, select vision backbones and input resolutions, tune algorithmic hyperparameters, and dictate execution order to prevent thermal or memory accumulation bias.

This module details how benchmark parameters are selected, parameterized algorithm variants are managed, and task queues are assembled prior to execution.

---

## 2. Configuration Options & Capabilities

### 2.1 Image Source Ingestion
The workspace supports three simultaneous input sources ([`gui/views/configure.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/configure.py)):
1. **Auto-Loaded Directory Images**: The dashboard automatically scans [`gui/images/`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/images) on startup, preloading all detected `.jpg`, `.jpeg`, `.png`, `.webp`, and `.gif` files.
2. **Interactive File Uploads**: Users can drag-and-drop multiple local image files directly into the UI.
3. **Remote Web URLs**: Users can enter newline-separated HTTP/HTTPS image links. Remote images are validated and fetched into memory via [`gui/utils/loader.py:load_image_from_url()`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/loader.py).

### 2.2 Input Resolution Selection
Users can benchmark vision models across 25 discrete resolution presets or enter custom dimensions (bounded by `config.min_input_size = 32px`):
```
32px, 64px, 96px, 128px, 160px, 192px, 224px (Default), 256px, 288px, 299px, 
320px, 352px, 384px, 416px, 448px, 480px, 512px, 576px, 640px, 704px, 
768px, 832px, 896px, 960px, 1024px
```
For CNN architectures, testing across this range reveals asymptotic runtime and memory scaling curves. For Vision Transformers (`vit-b-16`, `swin-t` from `config.fixed_size_models`), the engine enforces strict `224px` constraints.

### 2.3 Tunable Algorithm Hyperparameters
The framework supports granular hyperparameter tuning defined centrally in [`gui/config.py:method_configs`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/config.py#L30-L87):

- **Integrated Gradients**:
  - `n_steps`: Number of approximation steps along the path (Default: `50`, choices: `25`, `50`, `100`, `250`, `Custom`).
  - `internal_batch_size`: Mini-batch size for chunking Riemann summation steps to prevent VRAM Out-of-Memory (OOM) errors (Default: `2`).
  - `baseline_mode`: Reference tensor baseline (`Zeros (Black)`, `Ones (White)`, `Input Mean`).
- **Gradient SHAP**:
  - `n_samples`: Number of background samples to evaluate (Default: `10`).
  - `stdevs`: Standard deviation of Gaussian noise added to inputs (Default: `0.0001`).
  - `baseline_mode`: Baseline sampling distribution (`Zeros & Mean`, `Zeros Only`, `Ones Only`).
- **Occlusion**:
  - `sliding_window_shapes`: Square occlusion patch dimension in pixels (Default: `15`).
  - `strides`: Sliding step size in pixels (Default: `8`).
  - `occlude_color`: Occlusion fill color (`0` for black, `127` for gray, `255` for white).
- **LIME**:
  - `n_samples`: Number of perturbation samples to fit surrogate linear models (Default: `500`).
  - `perturbations_per_eval`: Evaluation batch size per forward pass (Default: `10`).
  - `n_segments`: SLIC superpixel segmentation count (Default: `50`).

### 2.4 Multi-Version Parameter Comparison
Researchers can evaluate multiple parameter variations of the same underlying algorithm side-by-side within a single execution batch (e.g., `Integrated_Gradients (25 steps)` vs `Integrated_Gradients (100 steps)`). The configuration engine dynamically indexes each variation (`method_name_1`, `method_name_2`) and preserves independent parameter state dictionaries.

---

## 3. Measurement Rigor Controls & Task Queuing Strategies

### 3.1 Measurement Rigor Parameters
- **Warm-up Runs** (Default: `3`): Pre-execution passes to prime PyTorch JIT caches, CUDA kernel contexts, and GPU clock states. Excluded from all statistical calculations.
- **Measured Repeats** (Default: `5`): Repeated attribution calls wrapped in hardware synchronization. Median values are reported in UI tables to filter out background operating system jitter.
- **Memory Profiling Runs** (Default: `1`): Dedicated isolated executions that track peak VRAM/RAM allocations without timing clock contamination. Setting to `0` disables memory profiling and displays `-`.

### 3.2 Task Queuing Strategies

```
┌─────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Strategy        │ Execution Sequencing Logic                                            │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Balanced        │ For each (Image, Model), rotates the XAI method order while keeping    │
│ (Default)       │ resolutions sorted small-to-large (32px → 1024px). Prevents thermal    │
│                 │ position penalties while providing rapid early UI feedback.            │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Sequential      │ Strict hierarchical nested order: Image → Model → Method → Resolution. │
│                 │ Evaluates all sizes for a given method before advancing.               │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ Randomized      │ Fully permutes every (Image, Model, Method, Resolution) task tuple     │
│                 │ using a deterministic seed (default: 42). Eliminates caching and       │
│                 │ thermal accumulation bias across long evaluation runs.                 │
└─────────────────┴────────────────────────────────────────────────────────────────────────┘
```

The queue construction is implemented in [`gui/utils/processing.py:build_task_queue()`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/processing.py) and ensures reproducible task schedules across GUI and CLI runs.

### 3.3 Explanation Quality & Fidelity Suite Configuration
Users can toggle post-hoc quality evaluation outside the timed benchmark loop. Supported metrics include:
- `Gini Index (Sparsity)`: Spatial concentration and focus.
- `Deletion AUC` & `Insertion AUC`: Faithfulness and sufficiency curves.
- `Sensitivity (Max)`: Robustness against subtle input jitter.
- `Infidelity (Perturbation Faithfulness)`: Logit perturbation mean squared error.

> [!WARNING]
> **Infidelity Incompatibility on Region Methods**: Infidelity evaluates fine-grained Gaussian perturbations ($\delta^T A(x)$). For region-based or superpixel explanation methods (defined in `config.region_based_methods = ["Occlusion", "LIME"]`), this metric is mathematically ill-suited and will produce distorted, invalid scores because coarse patch attributions cannot accurately track high-frequency pixel noise. Both the GUI and CLI actively inspect `config.region_based_methods` to issue contextual alerts.

---

## 4. Key Files & Directory Mapping

- [`gui/views/configure.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/configure.py) — Staging view UI, file uploaders, hyperparameter modal panels, and launch controls.
- [`gui/config.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/config.py) — Parameter definitions, options lists, and default constraints.
- [`gui/utils/processing.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/processing.py) — Task queue generation, method expansion, and resolution string parsers.
- [`gui/images/`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/images) — Auto-scanned local test image workspace folder.

---

## 5. Related Documentation

- [`01-architecture-and-lifecycle.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/01-architecture-and-lifecycle.md) — Describes the application state lifecycle and view routing.
- [`03-model-zoo-and-preprocessing.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/03-model-zoo-and-preprocessing.md) — Explains the underlying model zoo architectures configured here.
- [`05-benchmark-execution-engine.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/05-benchmark-execution-engine.md) — Details the execution loop that consumes the configured task queue.
