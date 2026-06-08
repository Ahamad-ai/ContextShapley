"""
Evaluation metrics for each benchmark type.

All evaluators return a float in [0.0, 1.0].
"""

import re
import string


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison (lowercase, strip articles/punct/whitespace)."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 score between prediction and gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    """Exact match after normalization. Returns 1.0 or 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def contains_answer(prediction: str, gold: str) -> float:
    """Check if the gold answer appears anywhere in the prediction."""
    return 1.0 if normalize_answer(gold) in normalize_answer(prediction) else 0.0


def gsm8k_evaluate(prediction: str, gold: str) -> float:
    """
    Evaluate GSM8K response.

    Extracts the number after #### or the last number in the response,
    then compares to gold.
    """
    # Try to find #### pattern
    match = re.search(r"####\s*(.+?)(?:\s|$)", prediction)
    if match:
        pred_num = match.group(1).strip().replace(",", "")
    else:
        # Fall back to last number in the response
        numbers = re.findall(r"-?\d+\.?\d*", prediction.replace(",", ""))
        if numbers:
            pred_num = numbers[-1]
        else:
            return 0.0

    gold_clean = gold.strip().replace(",", "")

    try:
        return 1.0 if float(pred_num) == float(gold_clean) else 0.0
    except ValueError:
        return 1.0 if pred_num == gold_clean else 0.0
