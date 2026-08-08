# Smart MCQ Solver

**DL & GenAI Project | IITM BS Diploma in Data Science and Applications**

Name: Om Manish Makadia  
Roll No: 23f3001593

An ensemble system that ranks the top-3 most likely correct answers for 5-option multiple-choice questions, built for the [Smart MCQ Solver Challenge](https://www.kaggle.com/) on Kaggle (evaluated on MAP@3).

**Public Leaderboard:** 0.75768 MAP@3 | Rank 320 / 1531 (Top 21%)  
**Live Demo:** https://mcq-solver-23365900968.asia-south1.run.app

---

## Overview

Three architecturally distinct models were trained and combined via a weighted ensemble:

| Model | Approach | Submission MAP@3 |
|---|---|---|
| Model 1 | Bi-LSTM + Attention over FastText embeddings (from scratch) | 0.7344 |
| Model 2 | DeBERTa-v3-base fine-tuned with LoRA (PEFT) | 0.7535 |
| Model 3 | Qwen2.5-3B-Instruct, QLoRA fine-tuned, with a custom RAG pipeline over a live Wikipedia corpus | 0.7510 |
| **Ensemble** | Test-score-informed weighted average of all three | **0.7577** |

Experiments were tracked end-to-end with Weights & Biases (versioned model artifacts via `wandb.Artifact`). Trained weights are hosted on the Hugging Face Hub (too large for git) and pulled at runtime by the deployed app.

---

## Repository Structure

```
dl-genai-project/
├── data/
│   ├── train.csv
│   ├── test.csv
├── notebooks/
│   ├── main.ipynb                       # EDA, training, and evaluation for all 3 models + ensembling
│   ├── rag_index_builder.ipynb          # Builds the live Wikipedia corpus + FAISS index for Model 3's RAG
├── reports/
│   └── report.docx
│   └── report.pdf
├── app.py                               # Gradio app entrypoint (deployment)
├── src/
│   ├── model1.py                        # Model 1 inference (BiLSTM + Attention)
│   ├── model2.py                        # Model 2 inference (DeBERTa-v3-LoRA)
│   ├── model3.py                        # Model 3 inference (Qwen2.5-3B + QLoRA + RAG)
│   └── ensemble.py                      # Combines the three models' probabilities
│   └── warm_cache.py                    # Pre-downloads HF model weights during the Docker build
├── Dockerfile
├── cloudbuild.yaml                      # Cloud Build config (used to build + push the image)
├── requirements.txt
└── README.md
```

Trained model weights live on the Hugging Face Hub:
- `23f3001593/dlgenai-model1-bilstm-attention-fasttext`
- `23f3001593/dlgenai-model2-deberta-v3-lora`
- `23f3001593/dlgenai-model3-qwen2p5-3b-qlora-rag`
- `23f3001593/dlgenai-ensemble-config`

---

## How to Run Locally

**1. Clone the repo and set up the environment**
```bash
git clone <repo-url>
cd dl-genai-project

conda create -n mcq-deploy python=3.10 -y
conda activate mcq-deploy
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```
(`requirements.txt` pins the CPU-only PyTorch build — no GPU is required to run the app.)

**3. Authenticate with Hugging Face**
The app pulls model weights from private HF Hub repos at startup, so you need a token with read access:
```bash
hf auth login
# paste your Hugging Face access token when prompted
```

**4. Run the app**
```bash
python app.py
```
The first run downloads and caches all model weights (~8GB total) before the UI opens — this can take a few minutes. Once ready, you'll see:
```
Running on local URL:  http://0.0.0.0:8080
```
Open **http://localhost:8080** in your browser.

---

## How to Host on Google Cloud Run

This is the exact command sequence used to deploy the live demo, from project setup through the final working deployment.

### 1. Project setup
```bash
gcloud config set project iitm-dlgenai-smart-mcq-solver

# Enable required APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# Create an Artifact Registry repo to hold the Docker image
gcloud artifacts repositories create mcq-solver \
  --repository-format=docker \
  --location=asia-south1
```

### 2. Store the HF token as a GCP secret
Needed because the Docker build downloads model weights from private Hugging Face repos (`warm_cache.py`), so the build step needs credentials:
```bash
echo -n "hf_xxxxxxxxxxxxxxxxxxxx" | gcloud secrets create hf-token --data-file=-

PROJECT_NUMBER=$(gcloud projects describe iitm-dlgenai-smart-mcq-solver --format='value(projectNumber)')

gcloud secrets add-iam-policy-binding hf-token \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding hf-token \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### 3. Build the image
```bash
gcloud builds submit --config cloudbuild.yaml .
```
`cloudbuild.yaml` builds the Docker image (baking in model weights via `warm_cache.py`, using the secret from step 2) and pushes it to Artifact Registry. Re-run this any time application code changes — not needed for memory/flag-only redeploys.

### 4. Deploy to Cloud Run
```bash
gcloud run deploy mcq-solver \
  --image asia-south1-docker.pkg.dev/iitm-dlgenai-smart-mcq-solver/mcq-solver/app:latest \
  --region asia-south1 \
  --platform managed \
  --memory 18Gi \
  --cpu 8 \
  --cpu-boost \
  --timeout 300 \
  --max-instances 1 \
  --allow-unauthenticated
```

**Flag notes:**
- `--memory 18Gi --cpu 8`: sized to fit all three models (Qwen2.5-3B in float16, DeBERTa, BiLSTM, plus the BGE embedder/reranker for RAG) simultaneously in memory.
- `--cpu-boost`: gives full CPU during container startup specifically, speeding up model loading.
- `--max-instances 1`: keeps total reserved memory within the project's regional quota (Cloud Run reserves roughly 2× the configured memory during revision rollouts).
- `--allow-unauthenticated`: makes the service publicly accessible without a Google login.

### 5. Get the live service URL
```bash
gcloud run services describe mcq-solver --region asia-south1 --format="value(status.url)"
```

---

## Tech Stack

Python, PyTorch, Hugging Face Transformers, PEFT (LoRA/QLoRA), Datasets, Sentence-Transformers, FAISS, Gradio, Scikit-learn, Pandas, NumPy, spaCy, Matplotlib, Seaborn, Weights & Biases, Docker, Google Cloud Run, Google Cloud Build, Google Artifact Registry, Google Secret Manager.

---

## Report

See [`report.pdf`](./reports/report.pdf) for the full technical report, including dataset/EDA details, architecture diagrams for all three models, training hyperparameters, and comparative analysis.