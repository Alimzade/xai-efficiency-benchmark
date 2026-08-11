# 01 - System Architecture & Application Lifecycle

## 1. Overview & Purpose

The GUI application is a web-based dashboard built on **Streamlit** that provides an interactive control center for benchmarking Explainable AI (XAI) attribution methods. It manages the full lifecycle of experimental runs: configuring multi-model, multi-method, and multi-resolution workloads; displaying live execution feedback and memory telemetry; analyzing trade-offs through interactive charts; persisting historical data; and generating exportable artifacts.

The design is completely self-contained within the `gui/` directory and avoids coupling to parent directory structures or external dependencies.

---

## 2. Core Architecture & Application Lifecycle

### 2.1 Initialization & Framework Pre-Warming
When a user launches the app (via `streamlit run main.py` or the launcher scripts), Streamlit executes [`gui/main.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/main.py). 

To prevent UI lag during initial package imports, the application presents a modern multi-stage splash screen ([`gui/assets/splash.html`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/splash.html)) that steps through five initialization phases:
1. **Load Frameworks**: Imports heavy scientific libraries (`pandas`, `torch`, `pyarrow`, `matplotlib`).
2. **Load XAI Core**: Initializes the singleton `SessionManager` and detects hardware availability from [`gui/config.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/config.py).
3. **System Check**: Verifies CUDA and Apple Silicon MPS device acceleration.
4. **Workspace Check**: Preloads views and registers route handlers.
5. **Launch Transition**: Sets `st.session_state.initialized = True` and triggers an instant rerun to mount the main dashboard.

### 2.2 Routing & Navigation Flow
Navigation is managed through a synchronized sidebar radio controller that dynamically routes between four core views:

```
                  ┌─────────────────────────────────┐
                  │        Sidebar Navigation       │
                  └────────────────┬────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
  │  Configure   │         │  Active Run  │         │   History    │
  │ (Workspace & │   ───►  │  (Execution  │   ───►  │ (Historical  │
  │ Parameters)  │         │  & Results)  │         │  Batches)    │
  └──────────────┘         └──────────────┘         └──────────────┘
                                   │
                                   ▼
                           ┌──────────────┐
                           │Documentation │
                           │(Taxonomy &   │
                           │Formulations) │
                           └──────────────┘
```

- **Configure View** ([`gui/views/configure.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/configure.py)): The benchmark staging ground. Manages workspace images, model architectures, XAI hyperparameter overrides, repeat counts, and task queuing strategies.
- **Active Run / Results View** ([`gui/views/active_run.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/active_run.py)): Renders live execution metrics, progress bars, active task cards, per-image heatmaps, and aggregate Pareto trade-off charts upon batch completion.
- **History View** ([`gui/views/history.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/history.py)): Browses, inspects, and downloads past benchmark sessions with one-click configuration restoration.
- **Documentation View** ([`gui/views/documentation.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/views/documentation.py)): In-app interactive taxonomy of algorithms, mathematical definitions, and model architectures.

### 2.3 State Management & Reactivity
Streamlit executes top-to-bottom on every user interaction. To prevent configuration loss or stale output contamination across reruns:
- Global state variables are initialized in [`gui/main.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/main.py#L127-L200) with safe fallback defaults.
- Batch configurations are frozen into `st.session_state` at execution start (`current_batch_methods`, `current_batch_models`, `current_batch_sizes`), isolating active and completed results from subsequent sidebar tweaks.
- Batch persistence is synchronized between in-memory `st.session_state` and disk JSON logs via [`gui/utils/state.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/state.py).

### 2.4 Styling & Dark Mode Theme
Custom styling is injected globally from [`gui/assets/styles.css`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/styles.css) at application startup. The stylesheet implements a cohesive dark theme featuring:
- Glassmorphism cards with subtle border glows (`rgba(255, 255, 255, 0.08)`).
- High-contrast typography and status indicators.
- Responsive CSS grid containers for heatmap figures and performance cards.

---

## 3. Key Files & Directory Mapping

- [`gui/main.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/main.py) — Application entrypoint, routing engine, and state initialization.
- [`gui/config.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/config.py) — Self-contained global constants (`PROJECT_ROOT`), default options, and method parameter schemas.
- [`gui/utils/state.py`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/utils/state.py) — Result group structuring and batch JSON disk synchronization.
- [`gui/assets/styles.css`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/styles.css) — Custom dashboard styling and UI theme tokens.
- [`gui/assets/splash.html`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/assets/splash.html) — HTML/CSS template for the 5-step preloader splash screen.

---

## 4. Related Documentation

- [`02-environment-and-hardware.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/02-environment-and-hardware.md) — Explains how hardware devices detected during lifecycle startup are profiled and mapped.
- [`04-benchmark-configuration.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/04-benchmark-configuration.md) — Details the configuration view rendered during the primary application state.
- [`05-benchmark-execution-engine.md`](file:///C:/Users/anara/projects/xai-efficiency-benchmark/gui/docs/05-benchmark-execution-engine.md) — Details how the active run page triggers and orchestrates backend execution.
