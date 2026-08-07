import os
import time
import gradio as gr
import pandas as pd
from src.model1 import Model1Predictor
from src.model2 import Model2Predictor
from src.model3 import Model3Predictor
from src.ensemble import Ensembler

LABELS = ["A", "B", "C", "D", "E"]

print("Loading Model 1 (BiLSTM + FastText)...")
model1 = Model1Predictor()

print("Loading Model 2 (DeBERTa-v3-LoRA, merged)...")
model2 = Model2Predictor()

print("Loading Model 3 (RAG + QLoRA-Qwen2.5-3B, merged) -- this is the slow one...")
model3 = Model3Predictor()

print("Loading ensemble config...")
ensembler = Ensembler()

print("All models loaded. Ready.")


def solve_mcq(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    if not prompt.strip() or not all([opt_a.strip(), opt_b.strip(), opt_c.strip(), opt_d.strip(), opt_e.strip()]):
        return "Please fill in the question and all five options.", None, None

    options = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d, "E": opt_e}

    t0 = time.time()
    probs1 = model1.predict(prompt, options)
    probs2 = model2.predict(prompt, options)
    probs3 = model3.predict(prompt, options)
    elapsed = time.time() - t0

    combined, top3 = ensembler.combine(probs1, probs2, probs3)

    result_text = f"**Top 3 (ranked): {' > '.join(top3)}**  \n_Inference time: {elapsed:.1f}s_"

    combined_label_dict = {LABELS[i]: float(combined[i]) for i in range(len(LABELS))}
    per_model_df = pd.DataFrame({
        "Option": LABELS,
        "Model 1 (BiLSTM)": [float(p) for p in probs1],
        "Model 2 (DeBERTa)": [float(p) for p in probs2],
        "Model 3 (Qwen+RAG)": [float(p) for p in probs3],
        "Ensemble": [float(p) for p in combined],
    })

    return result_text, combined_label_dict, per_model_df


with gr.Blocks(title="Smart MCQ Solver") as demo:
    gr.Markdown("# Smart MCQ Solver\nEnter a question and 5 options. The ensemble of 3 models predicts the top-3 most likely answers.")
    gr.Markdown("_Note: Model 3 uses retrieval + a 3B-parameter LLM running on CPU, so each submission can take a while. Please be patient._")

    with gr.Row():
        with gr.Column():
            prompt_box = gr.Textbox(label="Question / Prompt", lines=3, placeholder="Enter the question...")
            opt_a_box = gr.Textbox(label="Option A")
            opt_b_box = gr.Textbox(label="Option B")
            opt_c_box = gr.Textbox(label="Option C")
            opt_d_box = gr.Textbox(label="Option D")
            opt_e_box = gr.Textbox(label="Option E")
            submit_btn = gr.Button("Predict Top-3", variant="primary")

        with gr.Column():
            result_md = gr.Markdown(label="Result")
            combined_label = gr.Label(label="Ensemble Probabilities", num_top_classes=5)
            per_model_table = gr.Dataframe(label="Per-Model Breakdown")

    submit_btn.click(
        fn=solve_mcq,
        inputs=[prompt_box, opt_a_box, opt_b_box, opt_c_box, opt_d_box, opt_e_box],
        outputs=[result_md, combined_label, per_model_table],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)))