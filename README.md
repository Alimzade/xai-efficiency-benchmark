# XAI Efficiency Benchmark 🔍

A toolkit to measure and compare the runtime, memory, and energy overhead of popular explainable AI (XAI) methods on various models and datasets.

## Motivation 🎯

Explainable AI is crucial for trust and transparency in model decisions. However, many explanation techniques introduce significant computational overhead. For example, perturbation-based methods may require *hundreds or thousands of forward passes* to generate a single explanation ([OpenVINO™ Explainable AI Toolkit User Guide — OpenVINO™ XAI 1.1.0 documentation](https://openvinotoolkit.github.io/openvino_xai/stable/user-guide.html#:~:text=%2A%20Flexible%20,Cons)), and model-agnostic methods like SHAP can be *prohibitively slow* on large models ([Explainable artificial intelligence (XAI): from inherent explainability to large language models](https://arxiv.org/html/2501.09967v1#:~:text=Also%2C%20the%20computational%20overhead%20when,In%20addition%2C%20model)). This efficiency gap means some XAI methods are impractical for real-time or resource-constrained deployment. Balancing interpretability with computational efficiency is a known trade-off ([Do All AI Systems Need to Be Explainable?](https://ssir.org/articles/entry/do_ai_systems_need_to_be_explainable#:~:text=5.%20The%20trade,When)). This toolkit benchmarks a variety of XAI methods across models and datasets, quantifying their runtime, memory, and energy costs.

## Prerequisites 

- **Python:** 3.9 (added to `PATH`)
- **Git:** to clone this repository

## Setup Instructions (Windows 10/11) 

1. **Open PowerShell as Administrator.**
   
3. **Navigate** to your project folder:
   ```powershell
   cd C:\path\to\your_project
   ```
4. **Create** a Python 3.9 virtual environment:
   ```powershell
   python3.9 -m venv env39
   ```
5. **Activate** the environment:
   ```powershell
   .\env39\Scripts\activate
   ```
6. **Open** your code editor (e.g., VSCode, Cursor) and restart the integrated terminal if needed.
7. **Clone** this repository and `cd` into it:
   ```powershell
   git clone https://github.com/Alimzade/xai-efficiency-benchmark.git
   cd xai-efficiency-benchmark
   ```
8. **Install** dependencies in the terminal
   ```
   pip install captum memory_profiler opencv-python requests tensorflow ipykernel pandas scikit-image scikit-learn seaborn
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Dataset Download 🗃️

**For ImageNet:**
1. Obtain a Hugging Face access token at https://huggingface.co/settings/tokens
2. Insert the token into `datasets\imagenet_download.py` as instructed in that file.

Then download all datasets with (may take some time):

```powershell
python datasets\download.py
```
   

## Running Benchmarks 

Launch Jupyter and open one of the notebooks in the `benchmarks/` folder:

- **`benchmark.ipynb`** (recommended): select method, model, dataset, hardware, and run full suite.
- **`local_benchmark.ipynb`**: run one configuration per cell.
- **`LIME_benchmark.ipynb`**: tune perturbation settings (e.g., sample count).
- **`extended_benchmark.ipynb`**: includes global XAI methods and aggregated views.

Results (JSON, CSV, plots) will be saved in `experiment_results/`.

## Extending the Framework 

- **Add XAI methods:** place a new Python file in `xai_methods/` following existing templates.
- **Add models:** update scripts in `models/` folder to include loading logic for your model (architecture and weights).
- **Add datasets:** update loader scripts in `datasets/` (`data_loader.py` & `dataset_loader.py`) and update `download.py` if needed.


## Utilities 🛠️

The `utils/` folder contains helper modules for common tasks:

- `utils/benchmark_utils.py`: Functions to measure time, memory, and energy. Add your hardware here. 


## Output and Results 📊

- **Raw results**: After running benchmarks, raw output (e.g., runtime logs, memory usage) is saved in `experiment_results/`. Each experiment run creates timestamped subfolders.  
- **Figures**: Run `evaluation.ipynb` to generate figures and tables if needed. The `figures/` folder contains generated plots (PNG/HTML) comparing methods by runtime, memory, and energy. These visuals help quickly see the efficiency differences.  
- **Reproducing plots**: Example code for plotting is included in the `analysis/` scripts. You can modify and re-run them to customize or extend the analysis.  

By inspecting these results, researchers can identify which XAI methods are most practical under resource constraints and how to optimize their use.

## Citation
If you use this benchmark, please cite:

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
