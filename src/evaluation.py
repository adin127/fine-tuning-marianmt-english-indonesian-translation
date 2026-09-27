"""
Evaluate a fine-tuned MarianMT model against the original Helsinki-NLP model.
1. 5,000-sentence test subset stratified by English sentence length.
2. Evaluate the fine-tuned MarianMT 90:10 model.
3. Evaluate the original Helsinki-NLP MarianMT model.
4. Repeat evaluation after removing punctuation (including '?' and '-').
5. Report BLEU, n-gram precision, brevity penalty, and save predictions.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import torch
from nltk.util import ngrams
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

EN_PATH = Path("data/NeuLab-TedTalks.en-id.en")
ID_PATH = Path("data/NeuLab-TedTalks.en-id.id")

# Path to the fine-tuned MarianMT 90:10 model.
FINE_TUNED_MODEL = Path("models/FineTunedTransformer")

# Original pretrained MarianMT model used as the baseline.
BASE_MODEL = "Helsinki-NLP/opus-mt-en-id"

OUTPUT_DIR = Path("results")
N_TEST = 5_000
RANDOM_STATE = 42
MAX_LENGTH = 128


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------

def clean_text_keep_question_mark(text: str) -> str:
    """Lowercase and keep alphabetic characters, spaces, '-' and '?'."""
    text = text.lower()
    text = re.sub(r"[^a-z\s\-\?]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_text_letters_only(text: str) -> str:
    """Lowercase and keep only alphabetic characters and whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_parallel_data() -> pd.DataFrame:
    """Load and clean the NeuLab-TEDTalks English-Indonesian corpus."""
    with EN_PATH.open("r", encoding="utf-8") as file:
        english = file.read().splitlines()

    with ID_PATH.open("r", encoding="utf-8") as file:
        indonesian = file.read().splitlines()

    max_length = max(len(english), len(indonesian))
    english.extend([np.nan] * (max_length - len(english)))
    indonesian.extend([np.nan] * (max_length - len(indonesian)))

    data = pd.DataFrame({"english": english, "indonesian": indonesian})
    data = data.dropna(subset=["english", "indonesian"]).copy()

    data["english_text"] = data["english"].astype(str).apply(
        clean_text_keep_question_mark
    )
    data["indonesian_text"] = data["indonesian"].astype(str).apply(
        clean_text_keep_question_mark
    )

    data = data[["english_text", "indonesian_text"]].drop_duplicates().reset_index(drop=True)
    return data


def build_test_subset(data: pd.DataFrame, n_total: int = N_TEST) -> pd.DataFrame:
    """Create the same proportional sentence-length sample used in the notebook."""
    data = data.copy()
    bins = [0, 10, 30, float("inf")]
    labels = ["pendek", "sedang", "panjang"]

    data["english_length"] = data["english_text"].str.split().apply(len)
    data["indonesian_length"] = data["indonesian_text"].str.split().apply(len)
    data["length_category"] = pd.cut(
        data["english_length"], bins=bins, labels=labels, right=False
    )

    distribution = data["length_category"].value_counts(normalize=True)
    sample_sizes = (distribution * n_total).round().astype(int)

    difference = n_total - sample_sizes.sum()
    if difference != 0:
        sample_sizes[sample_sizes.idxmax()] += difference

    samples = []
    for category, n in sample_sizes.items():
        samples.append(
            data[data["length_category"] == category].sample(
                n=int(n), random_state=RANDOM_STATE
            )
        )

    subset = pd.concat(samples).reset_index(drop=True)
    return subset[["english_text", "indonesian_text"]]


# -----------------------------------------------------------------------------
# Translation and evaluation
# -----------------------------------------------------------------------------

def load_model(model_name: str | Path):
    """Load a MarianMT tokenizer and model."""
    model_name = str(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model


def translate(
    sentences: list[str],
    tokenizer: MarianTokenizer,
    model: MarianMTModel,
) -> list[str]:
    """Translate a list of sentences using the available CPU/GPU device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for sentence in tqdm(sentences, desc="Translating"):
            encoded = tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            translated = model.generate(**encoded)
            prediction = tokenizer.decode(
                translated[0], skip_special_tokens=True
            )
            predictions.append(prediction)

    return predictions


def compute_ngram_precision(reference: str, prediction: str, n: int) -> float:
    """Compute clipped n-gram precision for one sentence."""
    ref_tokens = reference.split()
    pred_tokens = prediction.split()

    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 0.0

    ref_counter = Counter(ngrams(ref_tokens, n))
    pred_counter = Counter(ngrams(pred_tokens, n))

    matches = sum(
        min(pred_counter[gram], ref_counter.get(gram, 0))
        for gram in pred_counter
    )
    total = sum(pred_counter.values())

    return matches / total if total else 0.0


def compute_corpus_ngram_precision(
    references: list[str], predictions: list[str], n: int
) -> float:
    """Compute clipped n-gram precision across the complete corpus."""
    total_matches = 0
    total_ngrams = 0

    for reference, prediction in zip(references, predictions):
        ref_tokens = reference.split()
        pred_tokens = prediction.split()

        if len(pred_tokens) < n or len(ref_tokens) < n:
            continue

        ref_counter = Counter(ngrams(ref_tokens, n))
        pred_counter = Counter(ngrams(pred_tokens, n))

        total_matches += sum(
            min(pred_counter[gram], ref_counter.get(gram, 0))
            for gram in pred_counter
        )
        total_ngrams += sum(pred_counter.values())

    return total_matches / total_ngrams if total_ngrams else 0.0


def evaluate_predictions(
    references: list[str], predictions: list[str]
) -> dict[str, float | list[float]]:
    """Calculate BLEU, n-gram precision and brevity penalty."""
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    reference_length = sum(len(ref.split()) for ref in references)
    prediction_length = sum(len(pred.split()) for pred in predictions)

    if prediction_length == 0:
        brevity_penalty = 0.0
    elif prediction_length > reference_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - (reference_length / prediction_length))

    return {
        "bleu": float(bleu.score),
        "unigram_precision": float(bleu.precisions[0]),
        "bigram_precision": float(bleu.precisions[1]),
        "trigram_precision": float(bleu.precisions[2]),
        "4gram_precision": float(bleu.precisions[3]),
        "reference_length": int(reference_length),
        "prediction_length": int(prediction_length),
        "brevity_penalty": float(brevity_penalty),
    }


def save_predictions(
    references: list[str],
    predictions: list[str],
    filename: str,
) -> pd.DataFrame:
    """Save reference/prediction pairs for inspection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(
        {
            "reference": references,
            "prediction": predictions,
        }
    )
    output.to_csv(OUTPUT_DIR / filename, index=False)
    return output


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def run_experiment(
    name: str,
    model_name: str | Path,
    test_data: pd.DataFrame,
    source_column: str = "english_text",
    target_column: str = "indonesian_text",
) -> dict[str, float | list[float]]:
    """Run inference, evaluate predictions, and save the prediction file."""
    tokenizer, model = load_model(model_name)

    sources = test_data[source_column].astype(str).tolist()
    references = test_data[target_column].astype(str).tolist()

    # Keep paired reference/prediction records aligned and remove empty pairs.
    valid_pairs = [
        (source, reference)
        for source, reference in zip(sources, references)
        if source.strip() and reference.strip()
    ]
    sources = [pair[0] for pair in valid_pairs]
    references = [pair[1] for pair in valid_pairs]

    predictions = translate(sources, tokenizer, model)
    metrics = evaluate_predictions(references, predictions)

    print(f"\n{name}")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(
        "N-gram precision: "
        f"{metrics['unigram_precision']:.4f}, "
        f"{metrics['bigram_precision']:.4f}, "
        f"{metrics['trigram_precision']:.4f}, "
        f"{metrics['4gram_precision']:.4f}"
    )
    print(f"Brevity Penalty: {metrics['brevity_penalty']:.4f}")

    filename = name.lower().replace(" ", "_") + "_predictions.csv"
    save_predictions(references, predictions, filename)

    metrics["model"] = name
    return metrics


def main() -> None:
    """Run all evaluation experiments from the original notebook."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NeuLab-TEDTalks data...")
    data = load_parallel_data()
    print(f"Cleaned dataset: {len(data):,} sentence pairs")

    test_data = build_test_subset(data)
    test_data.to_csv(OUTPUT_DIR / "neulab_test_subset.csv", index=False)
    print(f"Evaluation subset: {len(test_data):,} sentence pairs")

    # ------------------------------------------------------------------
    # Experiment 1: Fine-tuned MarianMT 90:10
    # ------------------------------------------------------------------
    fine_tuned_results = run_experiment(
        "fine_tuned_90_10",
        FINE_TUNED_MODEL,
        test_data,
    )

    # ------------------------------------------------------------------
    # Experiment 2: Original pretrained MarianMT baseline
    # ------------------------------------------------------------------
    baseline_results = run_experiment(
        "base_marianmt",
        BASE_MODEL,
        test_data,
    )

    # ------------------------------------------------------------------
    # Experiment 3: Fine-tuned model after removing punctuation.
    # This corresponds to the notebook's second preprocessing experiment.
    # ------------------------------------------------------------------
    punctuation_removed = test_data.copy()
    punctuation_removed["english_text"] = punctuation_removed["english_text"].apply(
        clean_text_letters_only
    )
    punctuation_removed["indonesian_text"] = punctuation_removed[
        "indonesian_text"
    ].apply(clean_text_letters_only)

    punctuation_results = run_experiment(
        "fine_tuned_punctuation_removed",
        FINE_TUNED_MODEL,
        punctuation_removed,
    )

    # Save a compact comparison table for the portfolio README.
    comparison = pd.DataFrame(
        [fine_tuned_results, baseline_results, punctuation_results]
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\nEvaluation complete.")
    print(comparison[["model", "bleu", "unigram_precision", "bigram_precision", "trigram_precision", "4gram_precision"]].to_string(index=False))


if __name__ == "__main__":
    main()
