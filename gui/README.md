# XAI Efficiency Benchmark - GUI & CLI

**Motivation:** Explainable AI is crucial for trust and transparency in model decisions. However, many explanation techniques introduce significant computational overhead. For example, perturbation-based methods may require *hundreds or thousands of forward passes* to generate a single explanation ([OpenVINO™ Explainable AI Toolkit User Guide](https://openvinotoolkit.github.io/openvino_xai/stable/user-guide.html#:~:text=%2A%20Flexible%20,Cons)), and model-agnostic methods like SHAP can be *prohibitively slow* on large models ([Explainable artificial intelligence (XAI): from inherent explainability to large language models](https://arxiv.org/html/2501.09967v1#:~:text=Also%2C%20the%20computational%20overhead%20when,In%20addition%2C%20model)). This efficiency gap means some XAI methods are impractical for real-time or resource-constrained deployment. Balancing interpretability with computational efficiency is a known trade-off ([Do All AI Systems Need to Be Explainable?](https://ssir.org/articles/entry/do_ai_systems_need_to_be_explainable#:~:text=5.%20The%20trade,When)).

This toolkit provides the primary interface for balancing that interpretability trade-off. It offers both a rich **Streamlit Web UI** and a headless **Command Line Interface (CLI)** to evaluate multiple model architectures, input sizes, and XAI methods on both local images and remote URLs. It generates **attribution runtimes**, **peak memory usage**, **estimated energy consumption**, and **quality metrics**, all neatly documented and plotted into **CSV/PDF exports** and heatmap visual summaries.

---

## 1️⃣ Software Stack

*   **App Framework:** Streamlit (interactive UI & routing)
*   **Deep Learning:** PyTorch (v2.x)
*   **XAI Library:** Captum (Saliency, Integrated Gradients, Guided Backprop, Gradient SHAP, DeepLift, Grad-CAM, etc.)
*   **Data Processing:** Pandas, NumPy
*   **Image Processing:** OpenCV (`opencv-python`), scikit-image (SLIC superpixels)
*   **Visualizations:** Matplotlib, Seaborn (high-quality statistical scatter plots, bar charts, and Pareto frontiers)
*   **System Profiling:** `memory_profiler` (system RAM tracking for CPU runs)
*   **Exports:** ReportLab (automated PDF report compiler)
*   **CLI Tools:** `questionary` (interactive terminal workflows)

---

## 2️⃣ Project Structure & File Roles

The GUI files are structured as follows:

```text
├── gui/
│   ├── main.py             # Main dashboard entrypoint: handles routing and UI initialization
│   ├── config.py           # Global state and constants manager
│   ├── requirements.txt    # GUI-specific Python dependencies
│   ├── App.bat / app.sh    # Quick-launch executable scripts for the Streamlit dashboard
│   ├── analysis/           # Data aggregators and trade-off logic (pareto.py, metrics.py)
│   ├── backend/            # Execution engines and filesystem managers (benchmark_runner, quality_runner, session_manager, exporter)
│   ├── cli/                # Headless command-line benchmark runner scripts
│   ├── components/         # Reusable UI elements (cards, plots, media, tables)
│   ├── assets/             # Static datasets (docs_reference.json, amd/intel-cpus.csv)
│   ├── views/              # Individual dashboard views (configure, active_run, history, documentation)
│   ├── utils/              # Helper functions, parsers, and setup tools (setup_env.py, processing.py)
│   └── sessions/           # Created dynamically: stores task CSVs, PDF reports, and heatmaps
```

---

## 3️⃣ Quick Start

To launch the benchmark dashboard automatically:

*   **Windows:** Double-click the `App.bat` file in the `gui/` folder.
*   **Linux / macOS:** Run `./app.sh` in the `gui/` folder.
*   **CLI Runners:** `cli/Run_CLI.bat` (Windows) and `cli/run_cli.sh` (Linux/macOS)

### Manual Setup
If you prefer to configure your environment manually instead of using the automated launchers:
```bash
cd gui
python -m venv venv
source venv/bin/activate  # (Windows: .\venv\Scripts\activate)
python -m pip install --upgrade pip
python utils/setup_env.py # (Auto-installs PyTorch & requirements.txt)
python -m streamlit run main.py
```

---

## 4️⃣ Supported Models and Methods

**Architectures:**
*   **CNNs:** `resnet50`, `convnext-t`, `efficientnet-b0`, `mobilenet-v3-large`, `densenet121`
*   **Transformers:** `swin-t`, `vit-b-16`
*   **Hybrid / Optimized:** `regnet-y-8gf`

*(Note: Transformer-style models like `vit-b-16` and `swin-t` are strictly locked to `224px` input sizes.)*

**Captum / Backpropagation Methods:**
*   **Gradient-based:** `Saliency`, `Integrated_Gradients`, `Guided_Backprop`, `Input_X_Gradient`, `Gradient_Shap`
*   **Reference-based:** `DeepLift`, `DeepLift_Shap`
*   **Perturbation & Region-based:** `Occlusion`, `LIME`
*   **Activation-based:** `Grad_CAM`

**Tunable Hyperparameters:**
The benchmark natively supports testing parameterized variations of the same base XAI algorithm side-by-side (e.g., comparing `Integrated_Gradients` at `20` steps vs `50` steps). Users can configure these directly via the UI to explore speed/quality trade-offs.

---

## 5️⃣ Core Performance Metrics

The benchmark extracts the following primary computational metrics for every XAI method evaluated:

*   **Attribution Runtime**: The isolated execution time (in seconds) required to generate the attribution mask. The benchmark extracts the full statistical profile across measured repeats (median, mean, std dev, min, max) and reports the median in UI summaries to eliminate hardware measurement noise.
*   **Peak Memory Usage**: The maximum VRAM (GPU) or system RAM (CPU) footprint allocated during attribution generation (in MB). It is measured in dedicated profiling runs to avoid contaminating timing measurements.
*   **Estimated Energy**: The estimated electrical energy footprint (in kWh), derived from the median attribution runtime and the device's Thermal Design Power (TDP) to compare energy efficiency across hardware.

---

## 6️⃣ Explanation Quality & Faithfulness Suite

To complement timing and memory efficiency metrics, the framework includes a post-hoc evaluation suite to measure explanation accuracy and faithfulness. These metrics run **off the timing clock** to keep speed measurements clean:

*   **Gini Index (Sparsity)**: Measures the spatial focus and sharpness of the attribution maps. A score near `1.0` represents highly focused attribution, whereas a score near `0.0` indicates uniform blur.
*   **Deletion AUC**: Progressively masks the most important pixels (replacing them with a baseline like zero or mean) and measures the decay in prediction confidence. A lower Area Under the Curve (AUC) indicates a more faithful explanation.
*   **Insertion AUC**: Progressively introduces the most important pixels to a baseline blank image and measures the recovery of prediction confidence. A higher AUC indicates a more faithful explanation.
*   **Sensitivity (Max)**: Measures the maximum change (worst-case sensitivity) in the explanation when the input is subjected to slight perturbations. Lower values are better (more robust).

<br/>

*   **Infidelity**: Measures the mean-squared error (MSE) between the difference in model predictions (logits) under Gaussian perturbations and the dot product of the input perturbation with the attribution map. Lower values are better. **Note:** Computed in pixel space; for region-based methods (e.g., LIME), the metric scales unpredictably and may not be directly comparable to pixel-based methods.

---

## 7️⃣ Outputs & Visualizations

Every benchmark run generates a structured suite of visual and tabular results, displayed directly in the UI and exported to the session folder:

*   **Context & Reproducibility Logs**:
    *   **Full Hardware Specification**: Embedded hardware context (CPU/GPU names, TDP ratings, driver/software versions) ensuring the run's environment is fully documented.
    *   **Configuration Specification**: A dedicated metadata block mapping out the exact test parameters (number of images, vision model architectures, XAI methods, image resolutions, quality metrics, warmup runs, measured repeats, memory runs, task ordering strategy (with random seed if "Randomized"), and custom XAI hyperparameters) used during the execution.
*   **Per-Image Results (Granular Analysis)**: 
    *   **Heatmap Collage Figures**: For every evaluated image, the system renders side-by-side visual figure comparisons of attribution masks across all selected XAI methods and input resolutions.
    *   **Detailed Metrics Tables**: A granular data table accompanies each image collage, explicitly listing the runtime, peak memory, estimated energy, and quality scores for every method evaluated.
*   **Aggregate Analytics (Global Averages & Trade-offs)**:
    *   Once all images are processed, the UI unlocks a rich performance analysis dashboard featuring:
        *   **Efficiency Bar Charts**: Compare the mean attribution runtime across model architectures and input resolutions.
        *   **Runtime vs Memory Scatter Plots**: Visualize the computational footprint of each configuration.
        *   **Pareto Analysis Curves**: Scatter plots mapping trade-off frontiers between speed (runtime) and explanation quality (e.g., Deletion AUC, Sensitivity), automatically ranking the Pareto-optimal methods.
        *   **Configuration Averages (Tables)**: High-level summary tables computing the overall means for all recorded metrics.

*(Note: All benchmark results (whether executed via the UI or the CLI) are automatically stored in the `gui/sessions/` directory for offline analysis.)*

---

## 8️⃣ Precision Measurement Details

*   **Execution Counts**: Users have full control over the benchmark's rigor, by setting the number of **warm-up runs** (to prime GPU/CPU caches before timing), **measured repeats** (for stable median runtime calculations), and dedicated **memory runs** (isolated executions for tracking peak VRAM/RAM allocations).
*   **Timing & Memory Isolation**: Memory profiling is performed in dedicated executions (default: 1 run) separate from runtime benchmarks to avoid profiling overhead affecting speed measurements. Disabling memory profiling (setting the number of memory runs to 0) displays `–` instead of an incorrect 0.0 MB value.
*   **Benchmark Precision & Outliers**: Warm-up runs are excluded from reported statistics. Timed repetitions measure execution time using hardware synchronization (CUDA/MPS where applicable), and the median runtime is reported to reduce the influence of background system activity and transient performance spikes.
*   **Memory Tracking**: GPU memory (CUDA/MPS) is measured directly using PyTorch's memory allocators. CPU memory is sampled from system RAM every 100 ms, meaning brief allocation spikes during executions shorter than the sampling interval may not be captured.
*   **Cache Cleanup & OOM Prevention**: GPU memory caches (CUDA/MPS) are cleared between benchmark runs to improve isolation between methods. Memory-intensive attribution methods (e.g., Integrated Gradients) are executed in configurable mini-batches to reduce peak memory usage and minimise Out-of-Memory (OOM) failures.
*   **Task Execution Order & Seed Reproducibility**: The framework supports three distinct task queuing strategies to suit different benchmark requirements:
    *   `Balanced` *(Default)*: For each (image, model) pair, rotates the XAI method execution order while keeping input sizes sorted small-to-large (32px → 1024px). Method rotation ensures no single algorithm systematically suffers cold-cache or thermal position penalties across images, while small-to-large sizing delivers rapid early UI feedback.
    *   `Sequential`: Executes tasks in strict nested order (Image → Model → Method → Size), running all resolution sizes for a method before moving to the next without any rotation.
    *   `Randomized`: Permutes every configuration tuple (image, model, method, resolution) pseudo-randomly using a deterministic seed (`random.Random(seed)`, default: `42`), eliminating thermal and caching state accumulation bias across long evaluation runs while guaranteeing reproducible task sequences.
*   **Energy Estimation & TDP Limitations**: Estimated energy consumption is approximated as: `Energy (kWh) = Runtime (s) × TDP (W) / (3600 × 1000)`, where Thermal Design Power (TDP) is used as a constant proxy for device power. Although vendor-specific interfaces (e.g., Intel RAPL, NVIDIA NVML, and AMD SMI) can expose hardware power or energy measurements on supported platforms, there is no universally available, standardised, low-overhead, cross-platform mechanism for accurately measuring real-time power consumption across all hardware. Furthermore, short-lived executions—particularly those completing in milliseconds—may finish before monitoring interfaces provide sufficiently precise measurements. Consequently, TDP is used as a consistent approximation to enable reproducible relative comparisons of estimated energy consumption and carbon footprint across different explainability methods and hardware configurations. This metric estimates relative energy usage rather than actual measured electrical energy consumption.
    *   *TDP Data Sources:* The bundled GPU and CPU Thermal Design Power databases (`gui/assets/`) are compiled from the open-source hardware datasets [painebenjamin/dbgpu](https://github.com/painebenjamin/dbgpu) and [felixsteinke/cpu-spec-dataset](https://github.com/felixsteinke/cpu-spec-dataset).

---

## 9️⃣ Benchmark Customization

You can easily modify and extend the benchmark to customize it for your specific research needs:

### 1. How to Add a Custom Deep Learning Model
1.  Open [gui/utils/model_loader.py](utils/model_loader.py).
2.  Add your model to the `MODEL_ZOO` dictionary at the top of the file to register it with the UI:
    ```python
    MODEL_ZOO = {
        # ... existing models ...
        'my_custom_model': (MyCustomModelClass, None) # Or specify weights
    }
    ```
3.  If your model requires custom instantiation (i.e. it doesn't take a standard `weights=` argument), locate `load_model(model_name)` and add a custom block *before* the `MODEL_ZOO` check:
    ```python
    if model_name == "my_custom_model":
        model = MyCustomModelClass(pretrained=True).to(device)
    ```
4.  *Optional*: If your model requires non-standard preprocessing, update the helper function `preprocess_image` in [gui/utils/model_loader.py](utils/model_loader.py).
5.  *Optional*: If your model is not trained on ImageNet, add its label mapping inside [gui/utils/label_utils.py](utils/label_utils.py).

### 2. How to Add a Custom XAI Method
1.  Open [gui/backend/benchmark_runner.py](backend/benchmark_runner.py).
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
4.  Add your method's key to the `xai_opts` list in [gui/config.py](config.py) and update the taxonomy in [gui/assets/docs_reference.json](assets/docs_reference.json).
5.  *Optional*: You can adjust existing XAI hyperparameters (such as the number of steps for Integrated Gradients or the baseline tensor values) directly inside the `get_attr()` helper function in [gui/backend/benchmark_runner.py](backend/benchmark_runner.py).

### 3. How to Auto-Load Local Test Images
To have your own set of local test images load automatically on startup:
*   Create a folder named `images` inside the `gui/` directory (i.e., `gui/images/`).
*   Drop your `.jpg`, `.jpeg`, `.png`, `.webp`, or `.gif` images inside that folder.
*   The dashboard will automatically scan this folder on launch and display all detected images in the workspace preview gallery, removing the need to upload them manually every time.
*   *Note*: Exclusions made via the GUI's **Auto-Loaded Folder Images** multiselect panel are session-based. Refreshing the browser page will re-include all images in `gui/images/`. To permanently exclude an image, remove it from the directory.

### 4. Task Execution Order & Seed Configuration
When configuring benchmark runs via the Streamlit UI or headless CLI (`gui/cli/cli.py`), you can specify the task order strategy using `--run-order {Balanced,Sequential,Randomized}` and set the random seed for reproducible task shuffling using `--random-seed <INTEGER>` (default: `42`).


