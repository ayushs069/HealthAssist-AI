"""
╔══════════════════════════════════════════════════════════════════╗
║          Comparative RAG Evaluation — Both Experiments          ║
╚══════════════════════════════════════════════════════════════════╝

Runs Experiment 1 and Experiment 2 back-to-back and produces a
side-by-side comparison table of all evaluation metrics.

Usage:
    python evaluate.py

Output:
    results/experiment_1_results.csv
    results/experiment_2_results.csv
    results/comparison_summary.csv   ← aggregated comparison
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd

# Import experiment runners
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import experiment_1
import experiment_2


METRICS = ["rouge1", "rouge2", "rougeL", "bleu", "bertscore", "accuracy"]
RESULTS_DIR = os.path.join(ROOT, "results")


def print_banner():
    print("\n" + "="*70)
    print("  🔬  RAG Comparative Evaluation — Healthcare Knowledge Assistant")
    print("="*70)


def print_comparison(summary_df: pd.DataFrame):
    """Pretty-print the side-by-side comparison table."""
    print(f"\n{'='*70}")
    print("  📊  SIDE-BY-SIDE METRIC COMPARISON")
    print(f"{'='*70}")

    # Column headers
    header = f"  {'Metric':<14} {'Experiment 1':>14} {'Experiment 2':>14} {'Winner':>10}"
    print(header)
    print("  " + "-" * 54)

    for _, row in summary_df.iterrows():
        metric = row["Metric"]
        e1     = row["Experiment_1"]
        e2     = row["Experiment_2"]
        winner = "Exp 2 ✅" if e2 > e1 else ("Exp 1 ✅" if e1 > e2 else "Tie  🤝")
        print(f"  {metric.upper():<14} {e1:>14.4f} {e2:>14.4f} {winner:>10}")

    print(f"{'='*70}")


def run():
    print_banner()

    # ── Run Experiment 1 ──────────────────────────────────────────────────
    print("\n\n🚀  Launching Experiment 1...\n")
    df1 = experiment_1.run()

    # ── Run Experiment 2 ──────────────────────────────────────────────────
    print("\n\n🚀  Launching Experiment 2...\n")
    df2 = experiment_2.run()

    # ── Aggregate comparison ──────────────────────────────────────────────
    rows = []
    for metric in METRICS:
        rows.append({
            "Metric":       metric,
            "Experiment_1": round(df1[metric].mean(), 4),
            "Experiment_2": round(df2[metric].mean(), 4),
        })

    summary_df = pd.DataFrame(rows)

    # Save comparison CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Print comparison table
    print_comparison(summary_df)

    # ── Config comparison ──────────────────────────────────────────────────
    print("\n  📋  CONFIGURATION COMPARISON")
    print(f"  {'Parameter':<22} {'Experiment 1':<30} {'Experiment 2':<30}")
    print("  " + "-"*82)
    comparisons = [
        ("Chunk Strategy",  "Recursive",                   "Token-based"),
        ("Chunk Size",      "500",                         "300"),
        ("Chunk Overlap",   "50",                          "30"),
        ("Embedding Model", "all-MiniLM-L6-v2",           "BAAI/bge-large-en-v1.5"),
        ("Embedding Dim",   "384",                         "1024"),
        ("Vector DB",       "FAISS",                       "ChromaDB"),
        ("LLM",             "LLaMA 3 (Ollama)",            "Gemma 2 (Ollama)"),
        ("Retrieval",       "Similarity Search",           "MMR"),
        ("Prompt Template", "Basic",                       "Detailed"),
        ("Top-k chunks",    "3",                           "4"),
    ]
    for label, e1_val, e2_val in comparisons:
        print(f"  {label:<22} {e1_val:<30} {e2_val:<30}")

    print(f"\n  💾  Comparison saved → {summary_path}\n")
    print("="*70)


if __name__ == "__main__":
    run()