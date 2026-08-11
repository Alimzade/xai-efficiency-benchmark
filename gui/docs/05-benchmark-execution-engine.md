# 05 - Benchmark Execution Engine, Timing Precision & Memory Profiling

## 1. Overview & Purpose

Measuring Explainable AI attribution performance requires strict isolation of computational phases. Naive benchmarks frequently suffer from asynchronous GPU execution skew, memory profiler timing contamination, model reload overhead, or duplicate attribution passes for visualization.

The execution engine in [`gui/backend/benchmark_runner.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/benchmark_runner.py) orchestrates the task pipeline to ensure high-precision, noise-free measurements of attribution runtime, peak memory footprints, and quality metrics.

---

## 2. Benchmark Execution Architecture

Each evaluation unit is executed via [`run_benchmark_task()`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/benchmark_runner.py#L225) following a strictly phased pipeline:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Execution Pipeline                             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 1. Model Retrieval & Cache  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 2. Target Class Inference   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 3. Warm-up Passes (Un-timed)│
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 4. Timed Attribution Runs   │
                    │   (CUDA Sync + Wall Clock)  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 5. Isolated Memory Runs     │
                    │  (Peak VRAM Allocator Stats)│
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 6. Visualization & Heatmaps │
                    │   (Attribution Reused)      │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │ 7. Quality Metrics Suite    │
                    │    (Off-the-Clock Analysis) │
                    └─────────────────────────────┘
```

---

## 3. High-Precision Measurement Protocols

### 3.1 CUDA Synchronization & Timing Isolation
PyTorch CUDA operations are dispatched asynchronously to the GPU driver. Measuring wall-clock time with standard Python timers without synchronization measures queue submission time rather than GPU kernel execution.

The engine wraps all timed attribution calls in hardware synchronization barriers:

```python
if device.type == "cuda":
    torch.cuda.synchronize()
elif device.type == "mps":
    torch.mps.synchronize()

t0 = time.perf_counter()
attr = get_attr()

if device.type == "cuda":
    torch.cuda.synchronize()
elif device.type == "mps":
    torch.mps.synchronize()
t1 = time.perf_counter()
```

Warm-up passes prime GPU clock frequencies and CUDA graph structures prior to recording. The full statistical distribution across repetitions is computed (Median, Mean, Standard Deviation, Min, Max), with the **median runtime** reported as the primary metric to neutralize operating system context-switch spikes.

### 3.2 Model Caching Mechanism
Instantiating deep vision backbones (e.g., loading 86M parameters for ViT) can take hundreds of milliseconds. To prevent initialization overhead from distorting benchmark batches, [`gui/backend/benchmark_runner.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/benchmark_runner.py#L53) maintains a global memory cache:

```python
MODEL_CACHE = {} # Key: (model_name, device_str)
```
When a task requests a model already cached for that device, the pre-warmed instance is reused immediately. The resulting cache hit status is logged alongside task metadata.

### 3.3 Isolated Memory Profiling
Memory profilers (like `memory_profiler`) sample process RSS memory at discrete intervals, which incurs CPU overhead and skews microsecond timing loops.

To resolve this, memory tracking runs **completely independently** from runtime timing passes:
- **CUDA VRAM Profiling**: Uses PyTorch's native memory manager (`torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated()`). This directly measures the true peak tensor memory allocated during attribution generation.
- **CPU RAM Profiling**: Employs background sampling (`memory_usage(..., max_usage=True)`) during dedicated memory execution passes.
- **Cache Clearing**: GPU caches are purged between runs (`torch.cuda.empty_cache()` and `gc.collect()`) to prevent allocation bleeding across methods.

### 3.4 Zero-Cost Visualization (Attribution Reuse)
The attribution tensor computed during the timed pass is preserved in memory and passed directly to the visualization routine ([`gui/backend/benchmark_runner.py:save_attribution_heatmap()`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/benchmark_runner.py)). The algorithm is **never executed twice** to render UI figures.

---

## 4. Active Run UI Integration

During live benchmarking, [`gui/views/active_run.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/active_run.py) presents real-time telemetry:
- **Active Task Progress**: Task index, overall percentage, and current configuration indicator.
- **Live Elapsed Wall Clock**: Interactive JavaScript timer tracking batch duration.
- **Dynamic Task Cards**: Instantaneous runtime, peak memory, and target classification feedback rendered via [`gui/components/cards.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components/cards.py).
- **Collapsible Per-Image Groups**: Heatmap figures and granular data tables updated on-the-fly as each task completes.

---

## 5. Key Files & Directory Mapping

- [`gui/backend/benchmark_runner.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/benchmark_runner.py) — Core benchmark execution engine, CUDA barrier synchronization, memory profiler, and Captum attribution wrappers.
- [`gui/views/active_run.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/active_run.py) — Active execution view, progress bars, and live result rendering.
- [`gui/backend/quality_runner.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/quality_runner.py) — Post-hoc explanation quality metric evaluation suite.

---

## 6. Related Documentation

- [`02-environment-and-hardware.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/02-environment-and-hardware.md) — Explains hardware discovery and the energy estimation equation.
- [`04-benchmark-configuration.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/04-benchmark-configuration.md) — Details the task queuing strategies consumed by the runner.
- [`06-explanation-quality-suite.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/06-explanation-quality-suite.md) — Documents the mathematical definitions and evaluation of quality metrics.
