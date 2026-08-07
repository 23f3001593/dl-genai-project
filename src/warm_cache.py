import os
from huggingface_hub import hf_hub_download, snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN")

print("Warming HF cache: ensemble config...")
hf_hub_download(
    repo_id="23f3001593/dlgenai-ensemble-config",
    filename="ensemble_config.json",
    token=HF_TOKEN,
)

print("Warming HF cache: model1 (BiLSTM checkpoint)...")
hf_hub_download(
    repo_id="23f3001593/dlgenai-model1-bilstm-attention-fasttext",
    filename="model1_best.pt",
    token=HF_TOKEN,
)

print("Warming HF cache: model2 (DeBERTa-v3-LoRA, merged)...")
snapshot_download(repo_id="23f3001593/dlgenai-model2-deberta-v3-lora", token=HF_TOKEN)

print("Warming HF cache: model3 (Qwen2.5-3B + RAG bundle)...")
snapshot_download(repo_id="23f3001593/dlgenai-model3-qwen2p5-3b-qlora-rag", token=HF_TOKEN)

print("Cache warmed. All model weights are now baked into this image layer.")