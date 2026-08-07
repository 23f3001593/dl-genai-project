import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import faiss
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, CrossEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL3_REPO = "23f3001593/dlgenai-model3-qwen2p5-3b-qlora-rag"

DEFAULT_LABELS = ["A", "B", "C", "D", "E"]
DEFAULT_PROMPT_TEMPLATE = (
    "{context_block}"
    "Question: {prompt}\n"
    "A. {A}\n"
    "B. {B}\n"
    "C. {C}\n"
    "D. {D}\n"
    "E. {E}\n"
    "Answer:"
)

DEFAULT_PREFIX_PHRASES = [
    "Pick the best possible answer:",
    "Select the most accurate option:",
    "Determine the correct option:",
    "Identify the correct statement:",
    "Choose the correct answer:",
]
DEFAULT_SUFFIX_PHRASES = [
    "among the listed options.",
    "based on the given context.",
    "from the following choices.",
    "carefully.",
]


def clean_query(prompt, prefix_phrases, suffix_phrases):
    q = str(prompt).strip()
    for p in prefix_phrases:
        if q.startswith(p):
            q = q[len(p):].strip()
            break
    for s in suffix_phrases:
        if q.endswith(s):
            q = q[: -len(s)].strip()
            break
    return q


class Model3Predictor:
    def __init__(self):
        local_dir = snapshot_download(repo_id=MODEL3_REPO)

        with open(f"{local_dir}/inference_config.json") as f:
            self.config = json.load(f)

        self.labels = self.config.get("labels", DEFAULT_LABELS)
        self.max_length = self.config.get("max_length", 512)
        self.prompt_template = self.config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)
        self.rag_config = self.config.get("rag", {"enabled": False})

        raw_ids = self.config["option_token_ids"]
        self.option_token_ids = {k: int(v) for k, v in raw_ids.items()}
        self.option_ids_tensor = torch.tensor([self.option_token_ids[l] for l in self.labels])

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float16,
            low_cpu_mem_usage=True,
        ).to(DEVICE)
        self.model.eval()

        if self.rag_config.get("enabled"):
            rag_bundle_dir = f"{local_dir}/{self.config.get('rag_bundle_dir', 'rag_bundle')}"
            rag_embedder_dir = f"{local_dir}/{self.config.get('rag_embedder_dir', 'rag_embedder')}"
            rag_reranker_dir = f"{local_dir}/{self.config.get('rag_reranker_dir', 'rag_reranker')}"

            self.rag_embedder = SentenceTransformer(rag_embedder_dir)
            self.rag_reranker = CrossEncoder(rag_reranker_dir)
            self.rag_index = faiss.read_index(f"{rag_bundle_dir}/rag_index.faiss")
            self.rag_passages = pd.read_json(f"{rag_bundle_dir}/rag_passages.jsonl", lines=True)

    def _retrieve_context(self, query: str) -> str:
        rag = self.rag_config
        clean_q = clean_query(
            query,
            DEFAULT_PREFIX_PHRASES,
            DEFAULT_SUFFIX_PHRASES,
        )
        instructed_q = rag["query_instruction"] + clean_q

        q_emb = self.rag_embedder.encode([instructed_q], normalize_embeddings=True).astype("float32")
        _, idxs = self.rag_index.search(q_emb, rag["retrieve_k"])
        candidate_idxs = [i for i in idxs[0] if i != -1]
        if not candidate_idxs:
            return ""

        candidates = self.rag_passages.iloc[candidate_idxs]["text"].tolist()
        rerank_pairs = [[clean_q, c] for c in candidates]
        rerank_scores = self.rag_reranker.predict(rerank_pairs)

        ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
        top_chunks = [c for c, _ in ranked[: rag["rerank_top_k"]]]
        return " ".join(top_chunks)[: rag["max_context_chars"]]

    def _build_prompt(self, prompt: str, options: dict) -> str:
        context_block = ""
        if self.rag_config.get("enabled"):
            context = self._retrieve_context(prompt)
            if context:
                context_block = f"Context: {context}\n\n"
        return self.prompt_template.format(
            context_block=context_block,
            prompt=prompt,
            A=options["A"], B=options["B"], C=options["C"], D=options["D"], E=options["E"],
        )

    def _qwen_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(DEVICE)
        with torch.no_grad():
            out = self.model(input_ids=input_ids)
        return out.logits[0, -1, :].float().cpu()

    @torch.no_grad()
    def predict(self, prompt: str, options: dict) -> np.ndarray:
        full_prompt = self._build_prompt(prompt, options)
        input_ids = self.tokenizer(
            full_prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        ).input_ids

        next_token_logits = self._qwen_forward(input_ids)
        option_logits = next_token_logits[self.option_ids_tensor]
        probs = F.softmax(option_logits, dim=-1).numpy()
        return probs