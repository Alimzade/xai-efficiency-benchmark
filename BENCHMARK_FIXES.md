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
| BF-012 | P2 | Not Started | Python/Torch setup expectations are inconsistent. | Align README, launcher, and smart setup around supported Python versions and CUDA wheel expectations. | Fresh setup path is documented and reproducible. |
| BF-013 | P2 | Completed | GUI exposes only a subset of available fast/local XAI methods. | Add compatible Captum/backprop methods to the frontend first; leave slow perturbation methods for a parameterized UI pass. | Syntax check passed; GUI now includes GradientShap, DeepLift, DeepLiftShap, and Grad-CAM for configured CNNs. |

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

## Suggested Implementation Order

1. BF-001 and BF-002 together, because timing and duplicate attribution are coupled.
2. BF-003, so memory metrics mean what they claim.
3. BF-004 and BF-005, so runtime experiments become statistically useful.
4. BF-006 and BF-007, to reduce noise and run-order bias.
5. BF-008, then rerun image-size analysis.
6. BF-009 through BF-012, to improve reproducibility and repo hygiene.
