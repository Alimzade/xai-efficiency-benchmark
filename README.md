# XAI Efficiency Benchmark

This project benchmarks the computational cost of explainable AI methods for image classifiers. The current primary workflow is the Streamlit GUI in `gui/app.py`; the older notebooks are still kept for deeper experiments and historical evaluations.

The GUI can compare multiple model architectures, input sizes, and XAI methods on uploaded images or image URLs. It reports attribution runtime, peak attribution memory, prediction metadata, heatmaps, batch summaries, CSV exports, and PDF reports.

## Current Status

- Primary app: `gui/app.py`
- Windows launcher: `Run_Benchmark.bat`
- Linux/macOS launcher: `run_benchmark.sh`
- Batch output folder: `gui/sessions/<batch_id>/`
- Historical notebook outputs: `experiment_results/`

Generated GUI sessions are local artifacts and should normally stay out of git.

## Supported Models and Methods

The GUI currently exposes these torchvision models:

- `resnet50`
- `convnext-t`
- `efficientnet-b0`
- `swin-t`
- `regnet-y-8gf`
- `mobilenet-v3-large`
- `densenet121`
- `vit-b-16`

The GUI currently exposes these local Captum/backpropagation methods:

- `Saliency`
- `Integrated_Gradients`
- `Guided_Backprop`
- `Input_X_Gradient`
- `Gradient_Shap`
- `DeepLift`
- `DeepLift_Shap`
- `Grad_CAM`

Some architectures have fixed input-size expectations. In the GUI, transformer-style models such as `vit-b-16` and `swin-t` are locked to `224px`.

## Setup

Python 3.9+ is required. Python 3.10 or 3.11 is recommended if you run into package compatibility issues.

Clone the repository:

```powershell
git clone https://github.com/Alimzade/xai-efficiency-benchmark.git
cd xai-efficiency-benchmark
```

### Windows

The simplest path is:

```powershell
.\Run_Benchmark.bat
```

The launcher will:

- find a compatible Python installation,
- create `venv/` if needed,
- run `smart_setup.py`,
- install dependencies from `requirements.txt`,
- launch the Streamlit GUI.

Manual setup is also possible:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
python smart_setup.py
python -m streamlit run gui/app.py --browser.gatherUsageStats=false
```

### Linux/macOS

```bash
chmod +x run_benchmark.sh
./run_benchmark.sh
```

Manual setup:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python smart_setup.py
python -m streamlit run gui/app.py --browser.gatherUsageStats=false
```

## CUDA and Torch

`smart_setup.py` attempts to choose an appropriate Torch installation:

- NVIDIA GPU: installs the CUDA 11.8 Torch wheel index.
- Apple Silicon: installs the standard Torch packages with MPS support where available.
- CPU-only systems: installs the standard CPU/universal Torch packages.

If GPU setup fails, install the Torch wheel that matches your system from the official PyTorch selector, then run:

```powershell
python -m pip install -r requirements.txt
```

## Running a Benchmark

1. Launch the GUI.
2. Select one or more model architectures.
3. Select input sizes for CNN-based models.
4. Select XAI methods.
5. Set warmup runs and measured repeats.
6. Upload images or paste image URLs.
7. Click **Start Multi-Model Benchmark**. The app prepares a fresh batch view, clears previous output, and then starts the run automatically.

During a run, the app shows the active task, elapsed batch time, progress, and completed results for the current batch. Previous batch summaries are hidden while a new batch runs.

## Measurement Semantics

The main runtime metric is attribution runtime. It measures the XAI attribution call, not image loading, preprocessing, prediction, heatmap rendering, file saving, or Streamlit UI time.

The app also reports:

- warmup runs: untimed repeats before measurement,
- measured runs: timed repeats used for the reported statistics,
- median, mean, standard deviation, minimum, and maximum attribution runtime,
- peak attribution memory,
- total batch wall time.

For CUDA runs, timing synchronizes CUDA before and after measured attribution work. CUDA memory uses Torch peak allocated memory. For MPS (Apple Silicon) runs, timing synchronizes MPS in the same way. CPU memory is process-memory based and should not be interpreted as GPU VRAM.

**CPU memory limitation:** CPU peak memory is measured by polling process memory at 100 ms intervals. If an attribution method completes in less than ~100 ms (common for fast methods like Saliency or Grad-CAM on small inputs), the profiler may miss the allocation spike entirely and report 0.0 MB. This is a known sampling limitation. The reported CPU memory values are most reliable for slower methods or larger input sizes where execution exceeds the polling window. CUDA and MPS memory measurements are not affected by this limitation because they use event-driven allocator tracking.

## Image-Size Studies

Image-size results are naturally noisy and should not be treated as exponential by default. For a useful size study:

- use the same images for every size,
- keep model and method fixed,
- use Balanced or Randomized task order,
- use enough measured repeats, such as 30-100 for fast methods,
- compare median/mean with variance instead of a single run.

The GUI includes an image-size scaling summary when multiple input sizes are present.

## Outputs

Each GUI batch is saved under:

```text
gui/sessions/<batch_id>/
```

Typical files include:

- `batch_results.json`
- per-task `config.json`
- generated heatmaps,
- exported CSV reports,
- exported PDF reports.

Older notebook workflows save results under `experiment_results/`.

## Notebook Workflows

The notebooks are still useful for reproducing older experiments:

- `benchmark.ipynb`: broad benchmark workflow,
- `local_benchmark.ipynb`: smaller local runs,
- `LIME_benchmark.ipynb`: LIME parameter experiments,
- `extended_benchmark.ipynb`: extended/global-method experiments,
- `evaluation.ipynb`: analysis and plots.

## Extending the Project

- Add local XAI methods under `xai_methods/`.
- Add or modify model loading in `models/model_loader.py`.
- Add dataset logic under `datasets/`.
- Update GUI method support in `gui/benchmark_runner.py` and `gui/app.py`.

## Citation

If you use this tool in your work, please cite:

```bibtex
@software{alimzade2025xai,
  author  = {Anar Alimzade},
  title   = {Efficiency Benchmark for XAI},
  year    = {2025},
  month   = {May},
  doi     = {10.5281/zenodo.15321812},
  version = {1.0.0}
}
```
