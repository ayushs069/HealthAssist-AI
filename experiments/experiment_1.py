"""
╔══════════════════════════════════════════════════════════╗
║         Experiment 1 — Baseline RAG Configuration       ║
╠══════════════════════════════════════════════════════════╣
║  Chunk Strategy : Recursive | Size: 500 | Overlap: 50   ║
║  Embedding      : all-MiniLM-L6-v2  (384-dim, HF)       ║
║  Vector DB      : FAISS  (in-memory)                    ║
║  LLM            : LLaMA 3  (Ollama — local)             ║
║  Retrieval      : Similarity Search                      ║
║  Prompt         : Basic                                  ║
╚══════════════════════════════════════════════════════════╝

Run:  python experiments/experiment_1.py
"""

import sys
import os

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd
from rag_pipeline import build_pipeline, generate_answer
from evaluation import evaluate_all

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CONFIG = {
    "name":             "Experiment 1 — Baseline (LLaMA3 + FAISS + MiniLM)",
    "pdf_path":         os.path.join(ROOT, "data", "documents", "healthcare.pdf"),
    "chunk_size":       500,
    "chunk_overlap":    50,
    "chunk_strategy":   "recursive",
    "embedding_model":  "all-MiniLM-L6-v2",          # 384-dim HuggingFace model
    "db_type":          "faiss",
    "llm_model":        "ollama:llama3",
    "retrieval_type":   "similarity",
    "prompt_template":  "basic",
    "k":                3,
}

DATA_PATH    = os.path.join(ROOT, "data", "qa_pairs", "test_data.csv")
RESULTS_DIR  = os.path.join(ROOT, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "experiment_1_results.csv")


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────
def run():
    print(f"\n{'='*62}")
    print(f"  🧪  {CONFIG['name']}")
    print(f"{'='*62}")

    # Build RAG pipeline
    vectorstore = build_pipeline(
        pdf_path        = CONFIG["pdf_path"],
        chunk_size      = CONFIG["chunk_size"],
        chunk_overlap   = CONFIG["chunk_overlap"],
        chunk_strategy  = CONFIG["chunk_strategy"],
        embedding_model = CONFIG["embedding_model"],
        db_type         = CONFIG["db_type"],
    )

    # Load QA test set
    df = pd.read_csv(DATA_PATH)
    print(f"\n📂 Test set loaded: {len(df)} questions\n")

    results = []

    for idx, row in df.iterrows():
        question     = row["Question"]
        ground_truth = row["Ground Truth"]

        print(f"[{idx+1}/{len(df)}] ❓  {question}")

        try:
            prediction, _docs, _prompt = generate_answer(
                vectorstore,
                question,
                llm_model       = CONFIG["llm_model"],
                retrieval_type  = CONFIG["retrieval_type"],
                prompt_template = CONFIG["prompt_template"],
                k               = CONFIG["k"],
            )
        except Exception as e:
            print(f"  ❌  LLM Error: {e}")
            prediction = ""

        metrics = evaluate_all(ground_truth, prediction)

        print(f"     📝  Pred  : {str(prediction)[:120]}...")
        print(f"     ✅  Truth : {ground_truth[:100]}")
        print(
            f"     📊  ROUGE-1={metrics['rouge1']:.3f} | "
            f"BLEU={metrics['bleu']:.3f} | "
            f"BERTScore={metrics['bertscore']:.3f} | "
            f"Acc={metrics['accuracy']}"
        )
        print()

        results.append({
            "Question":     question,
            "Ground Truth": ground_truth,
            "Prediction":   prediction,
            **metrics,
        })

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULTS_PATH, index=False)

    # ── Print summary ──────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  📊  FINAL RESULTS — {CONFIG['name']}")
    print(f"{'='*62}")
    for metric in ["rouge1", "rouge2", "rougeL", "bleu", "bertscore", "accuracy"]:
        avg = result_df[metric].mean()
        print(f"  {metric.upper():12s} : {avg:.4f}")

    print(f"\n  💾  Results saved → {RESULTS_PATH}")
    print(f"{'='*62}\n")
    return result_df


if __name__ == "__main__":
    run()