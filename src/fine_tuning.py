"""
MarianMT Fine-Tuning — 90:10 Train/Validation Split
The implementation preserves the original experiment logic:
- Helsinki-NLP/opus-mt-en-id
- 90:10 train/validation split
- random_state=100
- max sequence length = 128
- learning rate = 2e-5
- batch size = 64
- dropout = 0.3
- 3 training epochs
- early stopping patience = 3
- BLEU-based model selection
"""

import warnings
from pathlib import Path

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-id"
MODEL_NAME = MODEL_CHECKPOINT.split("/")[-1]

SOURCE_LANG = "en"
TARGET_LANG = "id"

ENGLISH = "en"
INDONESIAN = "id"
ENGLISH_TEXT = "english_text"
INDONESIAN_TEXT = "indonesian_text"
TRANSLATION = "translation"
LABELS = "labels"
INPUT_IDS = "input_ids"

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128  # Kept for documentation; original notebook uses MAX_INPUT_LENGTH for targets.

BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_TRAIN_EPOCHS = 3
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 3

RANDOM_STATE = 100
VALIDATION_SIZE = 0.10

DEFAULT_DATA_PATH = Path("data/dataTEDtalks.csv")
DEFAULT_OUTPUT_DIR = Path("models/FineTunedTransformer")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def postprocess_text(preds: list, labels: list) -> tuple:
    """Clean prediction and reference texts before BLEU evaluation."""
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def prep_data_for_model_fine_tuning(
    source_lang: list,
    target_lang: list,
) -> list:
    """Convert source and target text into translation dictionaries."""
    data_dict = {TRANSLATION: []}

    for source_text, target_text in zip(source_lang, target_lang):
        data_dict[TRANSLATION].append(
            {
                ENGLISH: source_text,
                INDONESIAN: target_text,
            }
        )

    return data_dict


def generate_model_ready_dataset(
    dataset: list,
    source: str,
    target: str,
    tokenizer: AutoTokenizer,
) -> list:
    """
    Tokenize source and target text and prepare records for MarianMT.

    This follows the original notebook's preprocessing logic, including
    truncation and padding at a maximum length of 128 tokens.
    """
    prepared_data = []

    for row in dataset:
        inputs = row[source]
        targets = row[target]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
        )

        model_inputs[TRANSLATION] = row

        # Preserve the original notebook's target-tokenization approach.
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=True,
            )

        model_inputs[LABELS] = labels[INPUT_IDS]
        prepared_data.append(model_inputs)

    return prepared_data


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(eval_preds: tuple) -> dict:
    """Compute SacreBLEU and average generated sequence length."""
    metric = evaluate.load("sacrebleu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
    )

    labels = np.where(
        labels != -100,
        labels,
        tokenizer.pad_token_id,
    )

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds,
        decoded_labels,
    )

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )

    result = {
        "bleu": result["score"],
    }

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id)
        for pred in preds
    ]

    result["gen_len"] = np.mean(prediction_lens)

    return {
        key: round(value, 4)
        for key, value in result.items()
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
):
    """Run the 90:10 MarianMT fine-tuning experiment."""

    data_path = Path(data_path)
    output_dir = Path(output_dir)

    # Load and remove rows containing missing values.
    dataset = pd.read_csv(data_path)
    df = dataset.dropna()

    # Split English source and Indonesian target text.
    X = df[ENGLISH_TEXT]
    y = df[INDONESIAN_TEXT]

    x_train, x_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("Training samples:", len(x_train))
    print("Validation samples:", len(x_val))

    # Load tokenizer.
    tokenizer = MarianTokenizer.from_pretrained(
        MODEL_CHECKPOINT
    )

    # Prepare translation dictionaries.
    training_data = prep_data_for_model_fine_tuning(
        x_train.values,
        y_train.values,
    )

    validation_data = prep_data_for_model_fine_tuning(
        x_val.values,
        y_val.values,
    )

    # Tokenize training and validation data.
    train_data = generate_model_ready_dataset(
        dataset=training_data[TRANSLATION],
        tokenizer=tokenizer,
        source=ENGLISH,
        target=INDONESIAN,
    )

    validation_data = generate_model_ready_dataset(
        dataset=validation_data[TRANSLATION],
        tokenizer=tokenizer,
        source=ENGLISH,
        target=INDONESIAN,
    )

    # Convert to Hugging Face Dataset objects.
    train_df = pd.DataFrame.from_records(train_data)
    validation_df = pd.DataFrame.from_records(validation_data)

    train_dataset = Dataset.from_pandas(train_df)
    validation_dataset = Dataset.from_pandas(validation_df)

    # Load pretrained MarianMT model.
    model = MarianMTModel.from_pretrained(
        MODEL_CHECKPOINT
    )

    # Apply dropout configuration used in the original experiment.
    model.config.dropout = DROPOUT
    model.config.attention_dropout = DROPOUT
    model.config.activation_dropout = DROPOUT

    # Configure training.
    model_args = Seq2SeqTrainingArguments(
        str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=3,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to=["none"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=model_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
        ],
    )

    # Fine-tune MarianMT.
    trainer.train()

    # Save the fine-tuned model.
    trainer.save_model(output_dir)

    print(f"Fine-tuned model saved to: {output_dir}")

    return trainer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_model()
