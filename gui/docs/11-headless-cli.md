# 11 - Headless & Interactive Command Line Interface (CLI)

## 1. Overview & Purpose

In addition to the Streamlit web dashboard, the framework provides a standalone **Command Line Interface (CLI)** located in [`gui/cli/`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/cli). 

The CLI enables automated benchmarking on headless compute clusters, continuous integration (CI) environments, and remote SSH sessions while maintaining 100% feature parity with the GUI—utilizing the same backend execution engine, precision timing protocols, and PDF/CSV report generation pipelines.

---

## 2. CLI Modes & Execution Workflows

[`gui/cli/cli.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/cli/cli.py) operates in two distinct execution modes:

```
                          ┌────────────────────────┐
                          │       cli/cli.py       │
                          └───────────┬────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
     ┌───────────────────────┐                 ┌───────────────────────┐
     │  Interactive Wizard   │                 │ Fully Headless Mode   │
     │ (questionary prompts) │                 │ (CLI Argument Flags)  │
     └───────────────────────┘                 └───────────────────────┘
```

### 2.1 Interactive Terminal Wizard Mode
When launched without arguments (or via [`gui/cli/Run_CLI.bat`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/cli/Run_CLI.bat) on Windows or [`gui/cli/run_cli.sh`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/cli/run_cli.sh) on Linux/macOS), the CLI launches a guided, terminal-based wizard powered by `questionary`:
- Multiselect checkboxes for vision model backbones.
- Interactive selection of XAI methods and parameter overrides (supporting multi-instance tuning).
- Dynamic image provider selection (scans `images/`, loads specific files, URLs, or batch `.txt` files) with instant image counting feedback (e.g. `Detected 5 image(s)`) and non-destructive retry looping if a directory or file contains no valid images.
- Resolution prompts with automatic 224px locking when Vision Transformers (`vit-b-16`, `swin-t` from `FIXED_SIZE_MODELS`) are selected.
- Hardware-aware device selection (presents only detected acceleration platforms: `cuda` on NVIDIA machines, `mps` on Apple Silicon, or auto-locks to `cpu`).
- Contextual TDP override inputs asked exclusively for the chosen device target (e.g. prompts only for GPU TDP if CUDA/MPS is selected, or only for CPU TDP if CPU is selected), with prefilled auto-detected values and database-fallback prompts.
- Warmup, repeat, and task order strategy prompts (`Balanced`, `Sequential`, `Randomized` with reproducible seed).
- Post-hoc Explanation Quality Metrics selection (`Gini Index (Sparsity)`, `Deletion AUC`, `Insertion AUC`, `Sensitivity (Max)`, and `Infidelity (Perturbation Faithfulness)`).

### 2.2 Fully Headless Flag-Driven Mode
For scripted, unattended execution, the CLI supports explicit command-line flags:

```bash
python cli/cli.py \
  --images images/sample.jpg \
  --models resnet50,mobilenet-v3-large \
  --methods Saliency,Integrated_Gradients,Grad_CAM \
  --input-sizes 224,448 \
  --warmups 3 \
  --repeats 5 \
  --memory-runs 1 \
  --run-order Balanced \
  --quality-metrics "Gini Index (Sparsity),Deletion AUC"
```

### 2.3 Comprehensive CLI Parameter Reference

| Flag | Type / Choices | Default | Description |
|:---|:---|:---|:---|
| `--images` | Comma-separated files, dirs, URLs | Auto-scans `gui/images/` | Test image files, folders, or remote URLs to evaluate. |
| `--models` | Comma-separated choices from `MODEL_ZOO` | `resnet50` | One or more vision model backbones. |
| `--methods` | Comma-separated choices from `xai_opts` | `Saliency,Integrated_Gradients` | XAI attribution algorithms to evaluate. |
| `--input-sizes` | Comma-separated integers ($\ge 32$, e.g. `112,224,448`) | `224` | Input image dimensions to benchmark (presets up to `1024px`). |
| `--device` | `auto`, `cuda`, `mps`, `cpu` | `auto` | Compute device to execute models and attributions on. |
| `--warmups` | Integer | `3` | Number of un-timed pre-execution passes. |
| `--repeats` | Integer | `5` | Number of timed attribution measurement repetitions. |
| `--memory-runs` | Integer | `1` | Number of dedicated peak memory profiling passes. |
| `--run-order` | `Balanced`, `Sequential`, `Randomized` | `Balanced` | Task execution sequencing strategy. |
| `--random-seed` | Integer | `42` | Deterministic random seed for task shuffling. |
| `--quality-metrics` | Comma-separated metric names | Disabled | Enables specified quality metrics (e.g. `'Gini Index (Sparsity),Deletion AUC'`). |
| `--cpu-tdp` | Float / Integer | Auto-lookup | Custom CPU TDP in Watts for energy estimation. |
| `--gpu-tdp` | Float / Integer | Auto-lookup | Custom GPU TDP in Watts for energy estimation. |
| `--config` | File path | `None` | Path to a JSON configuration file to execute directly. |


---

## 3. Parity with the Web Dashboard

The CLI shares the unified backend architecture with the Streamlit app:
- **Comprehensive Terminal Analytics**: Displays real-time task progress and mirrors the Web GUI's full analytical summary suite upon completion:
  1. `Method Parameters Mapping` (displayed directly in the startup configuration banner)
  2. `Configuration Averages` (runtimes, runtime std, peak memory, energy, and all evaluated quality metrics per method/model/size)
  3. `XAI Method Comparison` (rendered when multiple methods are benchmarked)
  4. `Pareto Analysis Summary` (non-dominated speed vs quality frontier rankings)
  5. `Model Comparison` (rendered when multiple architectures are benchmarked)
  6. `Resolution Comparison` (rendered when multiple input resolutions are benchmarked)
- **Identical Session Storage**: Results are saved into the identical [`gui/sessions/`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/sessions) directory structure.
- **GUI History Interoperability**: Batches generated via the CLI can be loaded, inspected, and visualized inside the Web UI's **History** tab with full interactive charts and tables.
- **Standardized Exports**: Generates the same high-fidelity ReportLab PDF reports (`report.pdf`) and standardized CSV summaries (`batch_summary.csv`).

---

## 4. Key Files & Directory Mapping

- [`gui/cli/cli.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/cli/cli.py) — Interactive wizard and headless CLI argument parser.
- [`gui/cli/Run_CLI.bat`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/cli/Run_CLI.bat) — Windows launcher batch script for the CLI.
- [`gui/cli/run_cli.sh`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/cli/run_cli.sh) — Linux / macOS launcher shell script for the CLI.

---

## 5. Related Documentation

- [`04-benchmark-configuration.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/04-benchmark-configuration.md) — Documents the configuration parameters and task queuing rules shared with the CLI.
- [`05-benchmark-execution-engine.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/05-benchmark-execution-engine.md) — Explains the core execution loop invoked by `cli.py`.
- [`10-export-systems.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/10-export-systems.md) — Details the ReportLab PDF and CSV generators executed at CLI batch completion.
