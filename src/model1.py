import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

MODEL1_REPO = "23f3001593/dlgenai-model1-bilstm-attention-fasttext"
CHECKPOINT_FILENAME = "model1_best.pt"

PAD, UNK = "<pad>", "<unk>"

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out, mask):
        scores = self.attn(lstm_out).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context


class BiLSTMEncoder(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix), freeze=False, padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)

    def forward(self, x):
        mask = (x != self.pad_idx).float()
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        context = self.attention(lstm_out, mask)
        return context


class MCQModel(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, pad_idx, dropout=0.4):
        super().__init__()
        self.encoder = BiLSTMEncoder(embedding_matrix, hidden_dim, pad_idx)
        combined_dim = hidden_dim * 2 * 2
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, q_ids, opt_ids):
        batch_size, num_opts, opt_len = opt_ids.shape
        q_context = self.encoder(q_ids)
        q_context_rep = q_context.unsqueeze(1).repeat(1, num_opts, 1)
        opt_flat = opt_ids.view(batch_size * num_opts, opt_len)
        opt_context = self.encoder(opt_flat)
        opt_context = opt_context.view(batch_size, num_opts, -1)
        combined = torch.cat([q_context_rep, opt_context], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        return logits


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s!?.,'\"]", "", text)
    text = re.sub(r"([!?.,'\"])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return text.split()

def encode(text, word2idx, max_len):
    ids = [word2idx.get(w, word2idx[UNK]) for w in tokenize(text)][:max_len]
    ids += [word2idx[PAD]] * (max_len - len(ids))
    return ids


class Model1Predictor:
    def __init__(self):
        ckpt_path = hf_hub_download(repo_id=MODEL1_REPO, filename=CHECKPOINT_FILENAME)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.word2idx = checkpoint["word2idx"]
        self.config = checkpoint["config"]
        self.labels = self.config["labels"]
        self.model = MCQModel(
            checkpoint["embedding_matrix"],
            self.config["hidden_dim"],
            pad_idx=self.word2idx[PAD],
            dropout=self.config.get("dropout", 0.4),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, prompt: str, options: dict) -> np.ndarray:
        q_text = clean_text(prompt)
        opt_texts = [clean_text(options[l]) for l in self.labels]
        q_ids = torch.tensor([encode(q_text, self.word2idx, self.config["max_len_question"])])
        opt_ids = torch.tensor(
            [[encode(o, self.word2idx, self.config["max_len_option"]) for o in opt_texts]]
        )
        logits = self.model(q_ids, opt_ids)
        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
        return probs