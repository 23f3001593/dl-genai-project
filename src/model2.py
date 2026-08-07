import json
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForMultipleChoice

MODEL2_REPO = "23f3001593/dlgenai-model2-deberta-v3-lora"

DEFAULT_LABELS = ["A", "B", "C", "D", "E"]

class Model2Predictor:
    def __init__(self):
        local_dir = snapshot_download(repo_id=MODEL2_REPO)
        cfg_path = f"{local_dir}/inference_config.json"
        try:
            with open(cfg_path) as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}
        self.labels = self.config.get("labels", DEFAULT_LABELS)
        self.option_cols = self.config.get("option_cols", DEFAULT_LABELS)
        self.max_length = self.config.get("max_length", 256)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.model = AutoModelForMultipleChoice.from_pretrained(local_dir, low_cpu_mem_usage=True)
        self.model.eval()

    @torch.no_grad()
    def predict(self, prompt: str, options: dict) -> np.ndarray:
        num_choices = len(self.option_cols)
        first_sentences = [prompt] * num_choices
        second_sentences = [options[col] for col in self.option_cols]
        tokenized = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.unsqueeze(0) for k, v in tokenized.items()}
        out = self.model(**inputs)
        probs = F.softmax(out.logits, dim=-1).squeeze(0).numpy()
        return probs