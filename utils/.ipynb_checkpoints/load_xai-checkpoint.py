import os
import importlib.util

def load_xai_method(xai_method):
    """
    Dynamically import the specified XAI method from subdirectories.

    Parameter:
    - xai_method: Name of the XAI method.

    Returns:
    - The imported XAI method module.

    Raises:
    - ModuleNotFoundError: If the XAI method is not found in the specified directories.
    """
    
    # Base directory containing the subdirectories with XAI methods
    base_dir = "xai_methods"
    sub_dirs = ["Local_Perturbation", "Local_Backpropagation", "Others"]

    for sub_dir in sub_dirs:
        method_path = os.path.join(base_dir, sub_dir, f"{xai_method}.py")  # Construct full path to module file
        if os.path.isfile(method_path):  # Check if the file exists
            # Load module from the specified path
            spec = importlib.util.spec_from_file_location(xai_method, method_path)
            method_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(method_module)
            print(f"Successfully loaded {xai_method} from {sub_dir}")
            return method_module

    raise ModuleNotFoundError(f"{xai_method} not found in any of the specified directories.")
