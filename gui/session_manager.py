import os
import uuid
from datetime import datetime
import json
import shutil

class SessionManager:
    def __init__(self, base_dir="gui/sessions"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def start_batch(self):
        """Generates a unique ID for a new benchmark batch."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = f"Batch_{timestamp}"
        batch_path = os.path.join(self.base_dir, batch_id)
        os.makedirs(batch_path, exist_ok=True)
        return batch_id

    def get_task_path(self, batch_id, img_idx, model_name):
        """Creates a subfolder for a specific task within a batch."""
        path = os.path.join(self.base_dir, batch_id, f"img{img_idx}_{model_name}")
        os.makedirs(path, exist_ok=True)
        return path

    def list_batches(self):
        """Lists only COMPLETED benchmark batches that have metadata."""
        if not os.path.exists(self.base_dir): return []
        batches = []
        for d in os.listdir(self.base_dir):
            if d.startswith("Batch_"):
                # Only include if the batch results file exists (proves completion)
                if os.path.exists(os.path.join(self.base_dir, d, "batch_results.json")):
                    batches.append({"id": d, "created": d.split("_")[1] + " " + d.split("_")[2]})
        return sorted(batches, key=lambda x: x["id"], reverse=True)

    def delete_batch(self, batch_id):
        """Deletes an entire batch."""
        path = os.path.join(self.base_dir, batch_id)
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
        return False

    def save_batch_config(self, batch_id, config):
        """Saves the configuration of a batch at the start of execution."""
        path = os.path.join(self.base_dir, batch_id, "batch_config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=4)

    def load_batch_config(self, batch_id):
        """Loads the configuration of an incomplete or complete batch."""
        path = os.path.join(self.base_dir, batch_id, "batch_config.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def list_incomplete_batches(self):
        """Lists batches that have a batch_config.json but do not have a batch_results.json."""
        if not os.path.exists(self.base_dir): return []
        incomplete = []
        for d in os.listdir(self.base_dir):
            if d.startswith("Batch_"):
                config_path = os.path.join(self.base_dir, d, "batch_config.json")
                results_path = os.path.join(self.base_dir, d, "batch_results.json")
                if os.path.exists(config_path) and not os.path.exists(results_path):
                    # Try to load some info from batch_config.json
                    try:
                        with open(config_path, "r") as f:
                            cfg = json.load(f)
                        num_images = len(cfg.get("image_sources", []))
                        num_models = len(cfg.get("models", []))
                        num_methods = len(cfg.get("methods", []))
                        info_str = f"{num_images} img, {num_models} mod, {num_methods} meth"
                    except Exception:
                        info_str = "Unknown config"
                    incomplete.append({
                        "id": d,
                        "created": d.split("_")[1] + " " + d.split("_")[2] if len(d.split("_")) > 2 else d,
                        "info": info_str
                    })
        return sorted(incomplete, key=lambda x: x["id"], reverse=True)

