# 02 - Environment Discovery, Hardware Profiling & Energy Estimation

## 1. Overview & Purpose

Reproducible XAI efficiency benchmarking requires rigorous characterization of the host computing environment. Attribution runtime and peak memory usage are heavily influenced by the underlying GPU architecture, driver levels, and CPU specifications. 

This module describes how the benchmark discovers host platform metadata, selects compute devices, looks up hardware Thermal Design Power (TDP) ratings, approximates electrical energy consumption, and provides automated smart environment provisioning.

---

## 2. Technical Mechanisms & Architecture

### 2.1 Hardware & Device Discovery
Hardware discovery is handled during application startup and task execution via [`gui/utils/helpers.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/helpers.py) and [`gui/backend/benchmark_runner.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/benchmark_runner.py):
- **CUDA GPUs**: Detects device availability via `torch.cuda.is_available()`, extracts the GPU device name (`torch.cuda.get_device_name(0)`), device capability, CUDA runtime version (`torch.version.cuda`), and total VRAM capacity.
- **Apple Silicon (MPS)**: Detects Metal Performance Shaders availability via `torch.backends.mps.is_available()`.
- **CPU Platform**: Extracts processor model names via Windows registry lookups (`HARDWARE\DESCRIPTION\System\CentralProcessor\0`), PowerShell CIM queries (`Get-CimInstance Win32_Processor`), Linux `/proc/cpuinfo`, or macOS `sysctl`.

### 2.2 Thermal Design Power (TDP) Lookup Databases
To enable energy estimation without requiring administrative kernel drivers or invasive hardware power meters, the framework bundles offline hardware specification databases located in [`gui/assets/`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets):
- **GPU Database (`dbgpu`)**: The `GPUDatabase` library maps NVIDIA, AMD, and Intel GPUs to their manufacturer rated TDP in Watts.
- **CPU Databases**: [`gui/assets/amd-cpus.csv`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/amd-cpus.csv) and [`gui/assets/intel-cpus.csv`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/intel-cpus.csv) match detected CPU model strings against comprehensive processor registries to extract baseline thermal power ratings.
- **Manual Overrides**: Users can override detected or missing TDP values directly from the UI or CLI if testing custom, overclocked, or under-volted hardware profiles.

### 2.3 Energy Estimation Model
Because millisecond-scale attribution operations often execute faster than hardware polling interfaces (e.g., RAPL or NVML) can sample without introducing measurement overhead, the benchmark employs a standardized TDP energy estimation model:

$$\text{Estimated Energy (kWh)} = \frac{\text{Attribution Runtime (s)} \times \text{Device TDP (W)}}{3600 \times 1000}$$

> [!NOTE]
> This metric provides a consistent, reproducible proxy for comparing relative electrical and carbon footprints across attribution algorithms and hardware targets without intrusive instrumentation overhead.

### 2.4 Smart Environment Setup (`setup_env.py`)
[`gui/utils/setup_env.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/setup_env.py) automates dependency management across operating systems:
1. Detects system hardware (NVIDIA GPU, Apple Silicon M-Series, or x86/ARM CPU).
2. Automatically installs the optimal PyTorch distribution (e.g., CUDA 11.8 wheels for NVIDIA cards to ensure broad legacy and modern GPU compatibility).
3. Installs remaining framework dependencies specified in [`gui/requirements.txt`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/requirements.txt).
4. Writes a `.setup_complete` indicator upon successful environment build.

---

## 3. Key Files & Directory Mapping

- [`gui/utils/setup_env.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/setup_env.py) — Auto-detects hardware and installs platform-optimized PyTorch binaries.
- [`gui/utils/helpers.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/helpers.py) — Hardware query utilities, CPU/GPU TDP database loaders, and device formatting helpers.
- [`gui/assets/amd-cpus.csv`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/amd-cpus.csv) — AMD processor catalog with TDP specifications.
- [`gui/assets/intel-cpus.csv`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/intel-cpus.csv) — Intel processor catalog with TDP specifications.
- [`gui/requirements.txt`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/requirements.txt) — GUI and backend Python dependency definitions.

---

## 4. Related Documentation

- [`01-architecture-and-lifecycle.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/01-architecture-and-lifecycle.md) — Explains how environment discovery runs during application initialization.
- [`05-benchmark-execution-engine.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/05-benchmark-execution-engine.md) — Details how device targets and memory allocators are synchronized during timing execution.
- [`10-export-systems.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/10-export-systems.md) — Documents how hardware metadata is formatted into exported PDF and CSV reports.
