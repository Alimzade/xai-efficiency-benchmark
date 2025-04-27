import torch
import tracemalloc

# Dictionary to store typical power usage for different devices (in watts) 
DEVICE_POWER_USAGE = {
    "RTX-2080-Ti": 250,
    "Tesla-T4": 70,
    "GTX-1050": 75,
    "P100": 250,
    "K80": 300,
    "Intel-Core-i7-9700K": 95,
    "AMD-Ryzen-5-3600": 65,
    "Intel-Xeon-E5-2650-v3": 105,
    "default-CPU": 65,
}

def get_device_name():
    """
    Retrieve the name of the first available device (GPU or CPU).
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def calculate_energy(avg_time, device_name=None, method=None, model=None):
    """
    Calculate energy consumption based on average time and device power usage.
    
    Parameters:
    - avg_time: Average time taken (in seconds).
    - device_name: Name of the device being used. If None, attempts to detect automatically.
    
    Returns:
    - Energy consumed in kWh and the device name used.
    """
    if device_name is None:
        device_name = get_device_name()
    
    power_usage = DEVICE_POWER_USAGE.get(device_name, 0)
    if power_usage == 0:
        print(f"Warning: Device '{device_name}' not found in power usage dictionary. Using default: 0W")
    
    print(f"Using...  Method: {method}, Model: {model}, Device: {device_name}, Power Usage: {power_usage}W")
    
    energy_kWh = avg_time * power_usage / 3600
    return energy_kWh, device_name

def measure_memory(image, predicted_class, generate_attribution, *args, **kwargs):
    """
    Measure peak and net memory usage for generating attributions.
    
    Parameters:
    - image: Input image tensor.
    - predicted_class: Predicted class for the input image.
    - generate_attribution: Function to generate attributions.
    - *args, **kwargs: Additional arguments required by generate_attribution.
    
    Returns:
    - Peak memory used (in MiB) and net memory change (in MiB).
    """
    device = image.device

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_usage_before = torch.cuda.memory_allocated()
        _ = generate_attribution(image, predicted_class, *args, **kwargs)
        mem_usage_after = torch.cuda.memory_allocated()
        peak_memory_used = torch.cuda.max_memory_allocated()
        peak_memory_used_mib = (peak_memory_used - mem_usage_before) / (1024 * 1024)
        net_memory_change_mib = (mem_usage_after - mem_usage_before) / (1024 * 1024)
    else:
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()
        _ = generate_attribution(image, predicted_class, *args, **kwargs)
        snapshot_after = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_memory_used_mib = peak / (1024 * 1024)
        stats_before = snapshot_before.statistics("traceback")
        net_memory_change_mib = 0 if not stats_before else (current - stats_before[-1].size) / (1024 * 1024)
    
    return peak_memory_used_mib, net_memory_change_mib

# Example usage for code testing
if __name__ == "__main__":
    avg_time = 120

    gpu_name = "RTX-2080-Ti"
    energy_gpu, device_gpu = calculate_energy(avg_time, device_name=gpu_name, method="ExampleMethod", model="ExampleModel")
    print(f"[GPU Test] Energy consumed: {energy_gpu:.6f} kWh for {device_gpu}")

    cpu_name = "Intel Core i7-9700K"
    energy_cpu, device_cpu = calculate_energy(avg_time, device_name=cpu_name, method="ExampleMethod", model="ExampleModel")
    print(f"[CPU Test] Energy consumed: {energy_cpu:.6f} kWh for {device_cpu}")

    unknown_device_name = "Unknown Device"
    energy_unknown, device_unknown = calculate_energy(avg_time, device_name=unknown_device_name, method="ExampleMethod", model="ExampleModel")
    print(f"[Unknown Device Test] Energy consumed: {energy_unknown:.6f} kWh for {device_unknown}")