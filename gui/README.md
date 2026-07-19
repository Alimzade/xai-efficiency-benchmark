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

## 🔬 Explanation Quality & Faithfulness Suite

To complement timing and memory efficiency metrics, the framework includes a post-hoc evaluation suite to measure explanation accuracy and faithfulness. These metrics run **off the timing clock** to keep speed measurements clean:

*   **Gini Index (Sparsity)**: Measures the spatial focus and sharpness of the attribution maps. A score near `1.0` represents highly focused attribution, whereas a score near `0.0` indicates uniform blur.
*   **Deletion AUC**: Progressively masks the most important pixels (replacing them with a baseline like zero or mean) and measures the decay in prediction confidence. A lower Area Under the Curve (AUC) indicates a more faithful explanation.
*   **Insertion AUC**: Progressively introduces the most important pixels to a baseline blank image and measures the recovery of prediction confidence. A higher AUC indicates a more faithful explanation.
*   **Infidelity**: Measures the scale-invariant mean-squared error (MSE) between the difference in model predictions under Gaussian perturbations and the dot product of the input perturbation with the attribution map. Lower values are better.

---

## 📂 Project Structure & File Roles

The GUI files are structured as follows:

```text
├── gui/
│   ├── app.py              # Main dashboard entrypoint: handles layout, pages, and plots
│   ├── benchmark_runner.py # Execution engine: manages warmups, repeats, timers, and VRAM
│   ├── docs_reference.json # Method, model, and metric reference taxonomy (JSON)
│   ├── exporter.py         # Report generator: compiles benchmark runs into CSV and PDF
│   ├── quality_runner.py   # Explanation quality & faithfulness evaluator: computes Gini, Deletion/Insertion AUC, and Infidelity
│   ├── session_manager.py  # File system coordinator: handles workspace outputs and cleanup
│   └── sessions/           # Created dynamically: stores task CSVs, PDF reports, and heatmaps
```

### 🔗 External Repository Dependencies

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
*   **Energy Estimation & TDP Limitations**: Estimated energy consumption is calculated as: `Energy (Wh) = Runtime (sec) * TDP (Watts) / 3600`. Modern computer hardware lacks standard, low-overhead, and cross-platform APIs to measure actual power consumption dynamically in real-time. Because of this, obtaining a direct, reliable reading of dynamic power draw for short-lived executions (especially sub-second runs that finish before sensors can sample) is not possible. Therefore, Thermal Design Power (TDP) serves as a standardized, constant proxy scaling factor to compare relative power and carbon footprints across varying hardware configurations.

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
