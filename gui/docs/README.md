# XAI Efficiency Benchmark — Technical Documentation Map

Welcome to the comprehensive technical documentation for the **XAI Efficiency Benchmark (GUI & CLI)** framework. This toolkit provides high-precision benchmarking, memory profiling, energy estimation, and explanation quality evaluation for Explainable AI (XAI) attribution methods across diverse deep learning vision models and input resolutions.

---

## 🗺️ Documentation Index & Reading Order

The documentation is organized sequentially from high-level system architecture and setup down to granular execution internals, analytical engines, export pipelines, and CLI tooling.

```
gui/docs/
├── README.md                              # Central documentation map & technical index (This file)
├── 01-architecture-and-lifecycle.md       # Application architecture, Streamlit lifecycle & session state
├── 02-environment-and-hardware.md         # Hardware discovery, CPU/GPU profiling & energy models
├── 03-model-zoo-and-preprocessing.md      # Vision backbones, weights, DeepLift patching & image pipeline
├── 04-benchmark-configuration.md          # Workspace setup, image loading, parameter tuning & task queuing
├── 05-benchmark-execution-engine.md       # Timing isolation, CUDA synchronization, model cache & memory runs
├── 06-explanation-quality-suite.md        # Quality metrics (Gini, AUC Deletion/Insertion, Sensitivity, Infidelity)
├── 07-analysis-and-pareto.md              # Statistical aggregation, trade-off curves & Pareto frontier ranking
├── 08-ui-components-and-visualizations.md # UI components, metric cards, Matplotlib charts & live progress
├── 09-session-management-and-history.md   # Batch persistence, JSON schema, history viewer & config restoration
├── 10-export-systems.md                   # Multi-page ReportLab PDF generation and CSV export architecture
├── 11-headless-cli.md                     # Interactive terminal workflow and automated headless CLI runner
└── 12-in-app-documentation-and-taxonomy.  # In-app reference guides, mathematical formulations & taxonomy
```

---

## 📑 Quick Topic Directory

| # | Document | Topic & Focus Area | Primary Source Files |
|:---|:---|:---|:---|
| **01** | [`01-architecture-and-lifecycle.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/01-architecture-and-lifecycle.md) | Streamlit app routing, initialization splash, state persistence, and CSS styling | [`main.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/main.py), [`config.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/config.py), [`utils/state.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/state.py) |
| **02** | [`02-environment-and-hardware.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/02-environment-and-hardware.md) | Platform detection, PyTorch device mapping, CPU/GPU TDP databases, and smart environment installation | [`utils/setup_env.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/setup_env.py), [`utils/helpers.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/helpers.py) |
| **03** | [`03-model-zoo-and-preprocessing.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/03-model-zoo-and-preprocessing.md) | Pretrained PyTorch models (CNNs, Transformers), ResNet ReLU patching for DeepLift, ImageNet labels, and image pipelines | [`utils/model_loader.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/model_loader.py), [`utils/label_utils.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/label_utils.py) |
| **04** | [`04-benchmark-configuration.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/04-benchmark-configuration.md) | Workspace image management (auto-load, upload, URLs), algorithm hyperparameters, and task queuing strategies (`Balanced`, `Sequential`, `Randomized`) | [`views/configure.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/configure.py), [`utils/processing.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/processing.py) |
| **05** | [`05-benchmark-execution-engine.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/05-benchmark-execution-engine.md) | Execution loop, warmup runs, CUDA synchronization, dedicated memory profiling runs, and model caching | [`backend/benchmark_runner.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/benchmark_runner.py), [`views/active_run.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/active_run.py) |
| **06** | [`06-explanation-quality-suite.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/06-explanation-quality-suite.md) | Mathematical formulation and evaluation logic for Gini Index, Deletion AUC, Insertion AUC, Sensitivity, and Infidelity | [`backend/quality_runner.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/quality_runner.py) |
| **07** | [`07-analysis-and-pareto.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/07-analysis-and-pareto.md) | Metrics aggregation across runs and 2D Pareto dominance frontier ranking (Runtime vs Quality/Memory) | [`analysis/metrics.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/analysis/metrics.py), [`analysis/pareto.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/analysis/pareto.py) |
| **08** | [`08-ui-components-and-visualizations.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/08-ui-components-and-visualizations.md) | Custom HTML/CSS stat cards, Matplotlib charts, Seaborn dark theme styling, live elapsed timer, and data tables | [`components/cards.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components/cards.py), [`components/plots.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components/plots.py), [`components/tables.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/components/tables.py) |
| **09** | [`09-session-management-and-history.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/09-session-management-and-history.md) | Filesystem session directory structure, `SessionManager`, JSON state persistence, and one-click historical configuration restoration | [`backend/session_manager.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/session_manager.py), [`views/history.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/history.py) |
| **10** | [`10-export-systems.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/10-export-systems.md) | Multi-page ReportLab landscape PDF generator (metadata, analytics, heatmaps, tables) and clean CSV exporter | [`backend/exporter.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/exporter.py) |
| **11** | [`11-headless-cli.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/11-headless-cli.md) | Interactive terminal wizard with `questionary` and headless CLI runner with argument parser and progress bars | [`cli/cli.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/cli/cli.py), [`cli/Run_CLI.bat`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/cli/Run_CLI.bat), [`cli/run_cli.sh`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/cli/run_cli.sh) |
| **12** | [`12-in-app-documentation-and-taxonomy.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/12-in-app-documentation-and-taxonomy.md) | In-app technical reference view and JSON taxonomy mapping for algorithms, backbones, and evaluation metrics | [`views/documentation.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/documentation.py), [`assets/docs_reference.json`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/docs_reference.json) |

---

## 🏗️ Architecture & Component Relationship Diagram

```
                              ┌────────────────────────┐
                              │     main.py (App)      │
                              └───────────┬────────────┘
                                          │
            ┌───────────────────┬─────────┴─────────┬───────────────────┐
            ▼                   ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │views/configure│   │views/active_rn│   │ views/history │   │  views/docs   │
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
            │                   │                   │                   │
            │                   ▼                   │                   │
            │           ┌───────────────┐           │                   │
            │           │backend/runner │           │                   │
            │           └───────┬───────┘           │                   │
            │                   │                   │                   │
            ▼                   ▼                   ▼                   ▼
    ┌───────────────────────────────────────────────────────────────────────────┐
    │  Shared Core: utils (loader, model_loader, processing, state, helpers)    │
    │  Analysis: analysis (metrics, pareto)  |  UI: components (cards, plots)   │
    │  Storage: backend/session_manager     |  Exports: backend/exporter        │
    └───────────────────────────────────────────────────────────────────────────┘
```
