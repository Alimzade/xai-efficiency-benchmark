# XAI Efficiency Benchmark GUI

An interactive, feature-rich Streamlit dashboard designed to configure, run, monitor, visualize, and export efficiency benchmarks for various Explainable AI (XAI) attribution methods across different Deep Learning model architectures, input resolutions, and hardware backends.

---

## 🛠️ Software Stack

*   **App Framework:** Streamlit (interactive UI & routing)
*   **Deep Learning:** PyTorch (v2.x)
*   **XAI Library:** Captum (Saliency, Integrated Gradients, Guided Backprop, Gradient SHAP, DeepLift, Grad-CAM, etc.)
*   **Data Processing:** Pandas, NumPy
*   **Visualizations:** Matplotlib, Seaborn (high-quality statistical boxplots and bar charts)
*   **System Profiling:** `memory_profiler` (system RAM tracking for CPU runs)
*   **Exports:** ReportLab (automated PDF report compiler)

---

## 🚀 Quick Start

To launch the benchmark dashboard:

*   **Windows:** Double-click the `Run_Benchmark.bat` file in the repository root.
*   **Linux / macOS:** Run `./run_benchmark.sh` in the repository root.
*   **Manual Launch:** Activate your virtual environment and execute:
    ```bash
    streamlit run gui/app.py
    ```

---

## 📂 Project Structure & File Roles

The GUI files are structured as follows:

```text
├── gui/
│   ├── app.py              # Main dashboard entrypoint: handles layout, pages, and plots
│   ├── benchmark_runner.py # Execution engine: manages warmups, repeats, timers, and VRAM
│   ├── exporter.py         # Report generator: compiles benchmark runs into CSV and PDF
│   ├── session_manager.py  # File system coordinator: handles workspace outputs and cleanup
│   └── sessions/           # Created dynamically: stores task CSVs, PDF reports, and heatmaps
```

### 🔗🔗 External Repository Dependencies

While the interface resides in the `gui/` folder, it strictly depends on the following external files and directories:

*   **`models/`** (Critical):
    *   `models/model_loader.py`: Handles target neural network architecture instantiation, preprocessing transformations, and caching.
    *   `models/label_utils.py`: Provides human-readable ImageNet label mappings for target prediction classes.
*   **`requirements.txt`** (Required):
    *   Defines python packages needed to install and execute the GUI workspace.
*   **`Run_Benchmark.bat` & `run_benchmark.sh`** (Root Launch Scripts):
    *   Bootstrappers in the root directory that automatically create the virtual environment, install requirements, and run the GUI via `streamlit run gui/app.py`.

---

## ⏱️ Precision Measurement Details

*   **Timing & Memory Isolation**: Memory profiling runs in dedicated executions (default 1) separate from speed tests to keep speed metrics clean. Disabling memory runs (setting to 0) displays a dash `–` instead of false 0.0 MB values.
*   **Benchmark Precision & Outliers**: Warmup runs are excluded from statistics. Timed repeats measure hardware duration using device synchronization (CUDA/MPS) and record median runtime to avoid background process spikes.
*   **Memory Tracking**: GPU memory (CUDA/MPS) is tracked directly by PyTorch allocators. CPU memory samples system RAM every 100 ms, which may miss short memory spikes for fast runs under 100 ms.
*   **Cache Cleanup & Crash Prevention**: Memory caches (CUDA/MPS) are cleared between runs to isolate methods. Heavy methods (like Integrated Gradients) run in small mini-batches to prevent Out-Of-Memory GPU crashes.

---

## 🔧 Benchmark Customization

You can easily modify and extend the benchmark to customize it for your specific research needs:

### 1. How to Add a Custom Deep Learning Model
1.  Open [models/model_loader.py](../models/model_loader.py).
2.  Locate `load_model(model_name)`.
3.  Add your custom instantiation block:
    ```python
    if model_name == "my_custom_model":
        model = MyCustomModelClass(pretrained=True)
        model.eval()
        return model.to(device)
    ```
4.  Add your model's name to the `SUPPORTED_MODELS` list in [gui/app.py](app.py).
5.  *Optional*: If your model requires non-standard preprocessing (e.g., custom normalization channels or mean/std values), update the helper function `preprocess_image` in [models/model_loader.py](../models/model_loader.py).

### 2. How to Add a Custom XAI Method
1.  Open [gui/benchmark_runner.py](benchmark_runner.py).
2.  Locate the XAI Tool Initialization block and instantiate your method:
    ```python
    elif method_key == 'my_custom_method':
        xai_tool = MyCustomMethodClass(model)
    ```
3.  Define the execution call inside the nested `get_attr()` helper function:
    ```python
    if method_key == 'my_custom_method':
        return xai_tool.attribute(input_tensor, target=pred_label_idx)
    ```
4.  Add your method's key to the `METHODS_INFO` list/dictionary in [gui/app.py](app.py).
5.  *Optional*: You can adjust existing XAI hyperparameters (such as the number of steps for Integrated Gradients or the baseline tensor values) directly inside the `get_attr()` helper function in [gui/benchmark_runner.py](benchmark_runner.py).

### 3. How to Auto-Load Local Test Images
To have your own set of local test images load automatically on startup:
*   Create a folder named `images` inside the `gui/` directory (i.e., `gui/images/`).
*   Drop your `.jpg`, `.jpeg`, `.png`, `.webp`, or `.gif` images inside that folder.
*   The dashboard will automatically scan this folder on launch and display all detected images in the workspace preview gallery, removing the need to upload them manually every time.
*   *Note*: Exclusions made via the GUI's **Auto-Loaded Folder Images** multiselect panel are session-based. Refreshing the browser page will re-include all images in `gui/images/`. To permanently exclude an image, remove it from the directory.
