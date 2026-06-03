import re
from rouge_score import rouge_scorer as rouge_scorer_module
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score_fn


# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────
# ROUGE (1, 2, L)
# ─────────────────────────────────────────────

def compute_rouge(reference: str, prediction: str) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.

    ROUGE measures n-gram overlap between prediction and reference.
        ROUGE-1 → unigram overlap
        ROUGE-2 → bigram overlap
        ROUGE-L → longest common subsequence
    """
    scorer = rouge_scorer_module.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    ref  = clean_text(reference)
    pred = clean_text(prediction)
    scores = scorer.score(ref, pred)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


# ─────────────────────────────────────────────
# BLEU
# ─────────────────────────────────────────────

def compute_bleu(reference: str, prediction: str) -> float:
    """
    Compute sentence-level BLEU score.

    BLEU measures precision of n-gram matches.
    Smoothing (method1) is applied to handle short predictions.
    """
    smoothie = SmoothingFunction().method1
    ref  = clean_text(reference).split()
    pred = clean_text(prediction).split()
    return round(sentence_bleu([ref], pred, smoothing_function=smoothie), 4)


# ─────────────────────────────────────────────
# BERTScore
# ─────────────────────────────────────────────

def compute_bertscore(reference: str, prediction: str) -> float:
    """
    Compute BERTScore F1 using contextual embeddings.

    BERTScore captures semantic similarity beyond surface overlap.
    Uses 'distilbert-base-uncased' by default (model=None, lang='en').
    """
    try:
        _, _, F1 = bert_score_fn(
            [str(prediction)], [str(reference)], lang="en", verbose=False
        )
        return round(F1.mean().item(), 4)
    except Exception as e:
        print(f"  ⚠️  BERTScore error: {e}")
        return 0.0


# ─────────────────────────────────────────────
# Accuracy (word-overlap based)
# ─────────────────────────────────────────────

def compute_accuracy(reference: str, prediction: str) -> int:
    """
    Binary accuracy: 1 if ≥30% word overlap, else 0.

    Heuristic metric for short factual answers.
    """
    ref  = set(clean_text(reference).split())
    pred = set(clean_text(prediction).split())
    if not ref:
        return 0
    overlap = len(ref & pred) / len(ref)
    return 1 if overlap >= 0.3 else 0


# ─────────────────────────────────────────────
# Combined Evaluation
# ─────────────────────────────────────────────

def evaluate_all(reference: str, prediction: str) -> dict:
    """
    Run all metrics and return a unified dict.

    Returns:
        {rouge1, rouge2, rougeL, bleu, bertscore, accuracy}
    """
    rouge = compute_rouge(reference, prediction)
    return {
        **rouge,
        "bleu":      compute_bleu(reference, prediction),
        "bertscore": compute_bertscore(reference, prediction),
        "accuracy":  compute_accuracy(reference, prediction),
    }