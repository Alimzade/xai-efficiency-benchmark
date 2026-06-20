# XAI Benchmark Fix Tracker

Use this file as the working checklist for benchmark-validity and runtime fixes.
Status should be one of: `Not Started`, `In Process`, `Completed`.

## Mandatory Fixes

| ID | Priority | Status | Issue | Implementation Notes | Verification |
| --- | --- | --- | --- | --- | --- |
| BF-001 | P0 | Completed | GPU timing is likely inaccurate because CUDA operations are asynchronous. | Synchronize CUDA before starting and after finishing timed attribution work, or use CUDA events. This must wrap the actual attribution call only. | Syntax check passed; benchmark repeat stability still needs empirical validation. |
| BF-002 | P0 | Completed | Attribution is computed twice in `gui/benchmark_runner.py`. | Store the attribution returned by the timed call and reuse it for heatmap generation. Do not call the XAI method again for visualization. | Syntax check passed; code now reuses the timed attribution for visualization. |
| BF-003 | P0 | Completed | CUDA memory measurement is not measuring real GPU VRAM. | Replace `memory_profiler` for CUDA with `torch.cuda.reset_peak_memory_stats()` and `torch.cuda.max_memory_allocated()`. Keep process RSS only for CPU if useful. | Syntax check passed; CUDA now reports Torch peak allocated memory for attribution. |
| BF-004 | P0 | Completed | Each image/model/size/method is measured once, so results are noisy. | Add configurable warmup count and measured repeat count. Report median, mean, std, min, and max runtime. For image-size comparisons, each size must be repeated many times, not just once. | Syntax check passed; empirical validation still needs a fresh benchmark batch with sufficient repeats. |
| BF-005 | P0 | Completed | Runtime semantics are unclear: method runtime, visualization time, and end-to-end task time are mixed in the UI narrative. | Separate attribution runtime from preprocessing, inference, visualization, saving, and Streamlit batch total. Label columns clearly. | Syntax check passed; UI, PDF, CSV, and config now distinguish attribution runtime from batch wall time. |
| BF-006 | P1 | Completed | Model is loaded for every benchmark step. | Cache/reuse loaded model objects per model/device during a batch, or restructure the runner to execute several methods/sizes per loaded model where safe. | Syntax check passed; runner now caches models per model/device and records cache status in results. |
| BF-007 | P1 | Completed | Run order can bias results through warmup/cache effects. | Add warmups and optionally randomized or balanced ordering for size/method/model combinations. | Syntax check passed; GUI now builds an explicit task queue with Balanced, Grouped, and reproducible Randomized order options. |
| BF-008 | P1 | Completed | Image-size analysis currently expects too much from noisy single-run results. | Treat size scaling as an experiment: same images, same model, same method, repeated trials per size, median aggregation, and variance/error bars. Avoid claiming exponential scaling; CNN compute often scales closer to pixel area. | Syntax check passed; GUI now reports image-size scaling with mean, standard deviation, sample count, and error-bar plot when multiple sizes are present. |
| BF-009 | P1 | Completed | ImageNet labels are fetched from GitHub during benchmark runs. | Bundle labels locally and load them without network dependency. | Syntax check passed; labels now come from local Torchvision weight metadata instead of GitHub. |
| BF-010 | P1 | Completed | Benchmark metadata is incomplete. | Save Python, Torch, CUDA availability/version, GPU name, device, input sizes, warmup/repeat counts, commit hash if available, and app version in `config.json`/batch metadata. | Syntax check passed; metadata appears in per-task config, batch metadata, and PDF reports. |
| BF-011 | P1 | Completed | Generated sessions and virtual environments pollute git status. | Update `.gitignore` for `venv/`, `venv-2/`, `gui/sessions/`, `__pycache__/`, `.streamlit/credentials.toml`, and generated result artifacts as appropriate. Do not delete user results without explicit request. | `.gitignore` now ignores future local environments, Streamlit credentials, caches, and GUI sessions; existing tracked session deletions still need to be committed or otherwise handled. |
| BF-012 | P2 | Completed | Python/Torch setup expectations are inconsistent. | Align README, launcher, and smart setup around supported Python versions and CUDA wheel expectations. | README now documents the GUI-first workflow, launchers, manual setup, CUDA/Torch behavior, outputs, and notebook legacy path; `run_benchmark.sh` now creates/activates `venv` before smart setup. |
| BF-013 | P2 | Completed | GUI exposes only a subset of available fast/local XAI methods. | Add compatible Captum/backprop methods to the frontend first; leave slow perturbation methods for a parameterized UI pass. | Syntax check passed; GUI now includes GradientShap, DeepLift, DeepLiftShap, and Grad-CAM for configured CNNs. |
| BF-014 | P1 | Completed | Live run output can show stale previous-batch results, and completion feedback can fire more than once. | Scope live/final result rendering and completion feedback to the active batch id. Build image result groups by image number instead of list position so Balanced/Randomized task order cannot mix rows. Replace balloons with a quieter completion status/toast. | Syntax check passed; new batches now use an auto-starting prepared state and a replaceable results placeholder before the engine starts so Streamlit can clear previous summaries. |
| BF-015 | P2 | Completed | GUI measurement controls and run state need clearer presentation. | Add compact run metrics, clearer measurement help, a calmer app header, and an auto-starting prepared-batch step before the benchmark engine starts. | Syntax check passed; visual validation still needs a manual Streamlit run. |
| BF-016 | P1 | Completed | Finished/current result views are coupled to the live sidebar selections. | Freeze batch methods, models, and input sizes when a run starts. Render current results and PDF exports from the batch's own method list, not the current sidebar state. | Syntax check passed; changing sidebar methods after a run no longer adds empty method rows or hides executed method rows. |
| BF-017 | P2 | Completed | Sidebar setting edits rerender the existing results panel. | Render sidebar settings in a Streamlit fragment when supported, so model/method/size edits can update naturally without forcing the right-side result view to rerun. Fall back to normal Streamlit behavior on older versions. | Syntax check passed; Apply Settings was removed. |

## Image-Size Analysis Notes

Current GUI results are fluctuative and should not be interpreted as a valid scaling curve yet.
For example, some `32 x 32` runs are slower than `224 x 224`, which suggests the benchmark is dominated by measurement noise, CUDA asynchrony, profiling overhead, reload/cache effects, or run-order effects.

Future image-size experiments should:

- Use the same image set for every size.
- Use the same model and XAI method for the comparison.
- Run warmups before measurements.
- Run many measured repeats for every size, such as 10, 30, or 100 depending on runtime.
- Report median plus variance/error bars, not a single runtime.
- Separate attribution runtime from visualization and file-saving time.
- Avoid claiming exponential scaling unless the measured curve supports it. For CNNs, a more natural first expectation is roughly related to pixel area, but architecture details and GPU kernels can make real curves non-linear.

## Remaining Work

The mandatory implementation fixes are complete. Remaining work is validation and optional polish:

1. Run a small manual GUI validation batch and confirm previous summaries clear after the prepare step.
2. Run a real image-size experiment with repeated measurements, then review whether the fluctuative pattern remains.
3. Review exported CSV/PDF output from a fresh batch for column order, labels, and metadata completeness.
4. Optionally add a more advanced non-blocking execution model later if the auto-starting prepared step is not robust enough on every browser.
