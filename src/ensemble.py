import json
import numpy as np
from huggingface_hub import hf_hub_download

ENSEMBLE_CONFIG_REPO = "23f3001593/dlgenai-ensemble-config"
ENSEMBLE_CONFIG_FILENAME = "ensemble_config.json"

class Ensembler:
    def __init__(self):
        cfg_path = hf_hub_download(repo_id=ENSEMBLE_CONFIG_REPO, filename=ENSEMBLE_CONFIG_FILENAME)
        with open(cfg_path) as f:
            self.config = json.load(f)
        self.labels = self.config["labels"]
        w = self.config["weights"]
        self.weights = {
            "model1": float(w.get("model1", 0.0)),
            "model2": float(w.get("model2", 0.0)),
            "model3": float(w.get("model3", 0.0)),
        }

    def combine(self, probs1: np.ndarray, probs2: np.ndarray, probs3: np.ndarray):
        combined = (
            self.weights["model1"] * probs1
            + self.weights["model2"] * probs2
            + self.weights["model3"] * probs3
        )
        combined = combined / combined.sum()
        ranked_idx = np.argsort(-combined)
        top3_labels = [self.labels[i] for i in ranked_idx[:3]]
        return combined, top3_labels