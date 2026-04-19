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
        """Lists all benchmark batches."""
        if not os.path.exists(self.base_dir): return []
        batches = []
        for d in os.listdir(self.base_dir):
            if d.startswith("Batch_"):
                batches.append({"id": d, "created": d.split("_")[1] + " " + d.split("_")[2]})
        return sorted(batches, key=lambda x: x["id"], reverse=True)

    def delete_batch(self, batch_id):
        """Deletes an entire batch."""
        path = os.path.join(self.base_dir, batch_id)
        if os.path.exists(path):
            shutil.rmtree(path)
            return True
        return False
