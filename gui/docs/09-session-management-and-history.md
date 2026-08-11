# 09 - Session Management, Storage Hierarchy & History Viewer

## 1. Overview & Purpose

Benchmark runs generate structured data arrays, high-resolution attribution heatmaps, execution logs, and exportable reports. To ensure full experimental reproducibility and facilitate offline review, the framework implements a persistent filesystem storage hierarchy.

This module details how benchmark batches are indexed and stored via [`gui/backend/session_manager.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/session_manager.py), browsed via [`gui/views/history.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/history.py), and restored for repeated experimentation.

---

## 2. Filesystem Storage Hierarchy

All benchmark executions (whether launched through the Streamlit Web GUI or the headless CLI) are automatically persisted under [`gui/sessions/`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/sessions) using unique, timestamped batch directories:

```
gui/sessions/
└── Batch_YYYYMMDD_HHMMSS/
    ├── batch_config.json    # Complete benchmark parameters and hyperparameter mappings
    ├── batch_results.json   # Full structured hierarchical execution records (loaded by GUI History)
    ├── batch_summary.csv    # Standardized flat presentation data table
    ├── report.pdf           # Multi-page ReportLab PDF export (collages + charts + tables)
    └── img<N>_<model>_<size>/
        ├── input_image.jpg  # Cached input image asset
        └── heatmaps/
            ├── Saliency.png
            ├── Integrated_Gradients.png
            └── Grad_CAM.png
```

---

## 3. Session Manager Architecture (`session_manager.py`)

The [`SessionManager`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/session_manager.py) class acts as the centralized filesystem manager:

- **Batch Initialization**: `start_batch()` allocates a timestamped directory and writes initial `batch_config.json`.
- **Image Subdirectories**: `get_task_path(batch_id, img_i, task_key)` creates localized task folders to store input images and attribution heatmaps without name collisions.
- **Session Enumeration**: `list_batches()` scans the `sessions/` directory, extracts batch metadata, and sorts sessions chronologically for presentation in the GUI History viewer.
- **Session Deletion**: `delete_batch(batch_id)` safely removes historical directories and frees up disk storage.

---

## 4. History View & One-Click Configuration Restoration

[`gui/views/history.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/history.py) provides an interactive archive interface:

```
                  ┌─────────────────────────────────────────┐
                  │          History Batch Browser          │
                  └────────────────────┬────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
     ┌──────────────────────┐                      ┌──────────────────────┐
     │  Inspect Historical  │                      │ Restore Config to UI │
     │  Batch Results & PDF │                      │  (One-Click Setup)   │
     └──────────────────────┘                      └──────────────────────┘
```

### 4.1 Historical Batch Inspection
Selecting a past batch loads its `results.json`, rendering the full analytical dashboard (summary cards, performance bar charts, Pareto frontiers, side-by-side heatmaps, and download buttons for PDF and CSV exports) identically to a live completed run.

### 4.2 One-Click Configuration Restoration
Reproducing or tweaking an earlier experiment is streamlined through the **"Restore Configuration"** action:
1. The historical `config.json` is loaded into `st.session_state.restore_config`.
2. The UI switches navigation to the **Configure** view.
3. The configuration page parses restored settings, populating the active models, resolutions, repeat counts, and custom XAI hyperparameters automatically.

---

## 5. Key Files & Directory Mapping

- [`gui/backend/session_manager.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/backend/session_manager.py) — Filesystem manager for creating, indexing, and purging session directories.
- [`gui/views/history.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/history.py) — History dashboard view, batch selector, and report download triggers.
- [`gui/sessions/`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/sessions) — Root storage directory for generated benchmark artifacts.

---

## 6. Related Documentation

- [`04-benchmark-configuration.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/04-benchmark-configuration.md) — Describes the configuration parameters restored from historical sessions.
- [`08-ui-components-and-visualizations.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/08-ui-components-and-visualizations.md) — Documents the component cards and plots re-rendered in the history viewer.
- [`10-export-systems.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/10-export-systems.md) — Explains the PDF and CSV files generated and stored within each session folder.
