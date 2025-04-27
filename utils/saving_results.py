import os
import json
import pandas as pd
from collections import OrderedDict

def get_next_run_number(method, model, dataset, gpu_name):
    """Determine the next run number for a given method, model, dataset, and GPU."""
    summary_file = "experiment_results/summary.jsonl"
    run_number = 1
    
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                if (entry.get("method") == method and
                    entry.get("model") == model and
                    entry.get("dataset") == dataset and
                    entry.get("gpu_name") == gpu_name):
                    run_number = max(run_number, entry.get("run_number", 0) + 1)
                    
    return run_number

def save_experiment_results(method, model, dataset, summary, per_image_results, avg_values, std_devs):
    # Ensure the `experiment_results` directory exists
    os.makedirs("experiment_results", exist_ok=True)
    
    # Determine the run number once
    run_number = get_next_run_number(method, model, dataset, summary.get("gpu_name", "unknown_gpu"))

    # Append to the summary file
    summary_file = "experiment_results/summary.jsonl"
    with open(summary_file, 'a') as f:
        ordered_summary = OrderedDict([
            ("method", method),
            ("model", model),
            ("dataset", dataset),
            ("num_images", summary.get("num_images")),
            ("gpu_name", summary.get("gpu_name")),
            ("run_number", run_number),
            ("average_time", avg_values["time"]),
            ("std_time", std_devs["time"]),
            ("average_peak_memory", avg_values["peak_memory"]),
            ("std_peak_memory", std_devs["peak_memory"]),
            ("average_net_memory_change", avg_values["net_memory_change"]),
            ("std_net_memory_change", std_devs["net_memory_change"]),
            ("energy_consumed_kWh", summary.get("energy_consumed_kWh"))
        ])

        # Write the ordered summary to file
        f.write(json.dumps(ordered_summary) + "\n")
    
    # Create the per_image_results directory if it doesn't exist
    per_image_results_dir = "experiment_results/per_image_results"
    os.makedirs(per_image_results_dir, exist_ok=True)
    
    # Prepare the file name based on parameters, including GPU name
    gpu_name = summary.get("gpu_name", "unknown_gpu").replace(" ", "_")
    per_image_results_file = f"{per_image_results_dir}/{method}_{model}_{dataset}_{gpu_name}_run{run_number}.csv"

    # Save per-image results to the CSV file
    per_image_df = pd.DataFrame(per_image_results)
    per_image_df["method"] = method
    per_image_df["model"] = model
    per_image_df["dataset"] = dataset
    per_image_df["run_number"] = run_number  # Use the same run number for per-image results
    
    per_image_df.to_csv(per_image_results_file, index=False)

    print(f"Results saved in: {per_image_results_file}")
