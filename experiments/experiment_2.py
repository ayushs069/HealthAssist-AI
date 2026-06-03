"""
╔══════════════════════════════════════════════════════════╗
║     Experiment 2 — Enhanced RAG Configuration           ║
╠══════════════════════════════════════════════════════════╣
║  Chunk Strategy : Token-based | Size: 300 | Overlap: 30 ║
║  Embedding      : BAAI/bge-large-en-v1.5  (1024-dim)    ║
║  Vector DB      : ChromaDB  (persistent)                ║
║  LLM            : Gemma 2  (Ollama — local)             ║
║  Retrieval      : MMR  (Maximal Marginal Relevance)      ║
║  Prompt         : Detailed role-prompt                   ║
╚══════════════════════════════════════════════════════════╝

Key differences vs Experiment 1:
  • Smaller chunks (300) for finer granularity
  • Larger embedding dimensions (1024 vs 384) → richer semantics
  • ChromaDB instead of FAISS
  • MMR retrieval → less redundant context
  • Detailed prompt → richer, more complete answers
  • Gemma 2 (Google) instead of LLaMA 3 (Meta)

Run:  python experiments/experiment_2.py
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
    "name":             "Experiment 2 — Enhanced (Gemma2 + ChromaDB + BGE-Large + MMR)",
    "pdf_path":         os.path.join(ROOT, "data", "documents", "healthcare.pdf"),
    "chunk_size":       300,
    "chunk_overlap":    30,
    "chunk_strategy":   "token",
    "embedding_model":  "BAAI/bge-large-en-v1.5",     # 1024-dim HuggingFace model
    "db_type":          "chroma",
    "persist_dir":      os.path.join(ROOT, "chroma_db_exp2"),
    "llm_model":        "ollama:gemma2",
    "retrieval_type":   "mmr",
    "prompt_template":  "detailed",
    "k":                4,
}

DATA_PATH    = os.path.join(ROOT, "data", "qa_pairs", "test_data.csv")
RESULTS_DIR  = os.path.join(ROOT, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "experiment_2_results.csv")


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
        persist_dir     = CONFIG.get("persist_dir"),
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