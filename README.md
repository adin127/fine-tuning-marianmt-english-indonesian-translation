# MarianMT-Based English–Indonesian Machine Translation with Domain-Specific Fine-Tuning

A Natural Language Processing (NLP) project that fine-tunes a pretrained Transformer-based MarianMT model for English-to-Indonesian machine translation using domain-specific TED Talks data. The project focuses on adapting a pretrained MarianMT model to formal English–Indonesian translation and evaluating how different train-validation split configurations and preprocessing strategies affect translation performance.

## Overview
Machine translation models are often trained on large and diverse datasets, but their performance can vary depending on the domain and characteristics of the input text. In this project, a pretrained **MarianMT** model was fine-tuned using English–Indonesian TED Talks data with a focus on formal language translation.
The project evaluates:
- The performance of pretrained MarianMT before fine-tuning
- The effect of domain-specific fine-tuning
- Different train-validation split configurations
- Training and validation performance across epochs
- The effect of question-mark preprocessing
- Translation performance using BLEU and n-gram precision

The highest documented BLEU score was **32.14** using a **90:10 train-validation split**.

## Project Objective
The main objective of this project is to investigate whether fine-tuning a pretrained MarianMT model using domain-specific TED Talks data can improve English-to-Indonesian translation performance.
More specifically, this project aims to:
1. Prepare and preprocess English–Indonesian parallel text data.
2. Fine-tune a pretrained MarianMT Transformer model.
3. Compare different train-validation split configurations.
4. Evaluate translation quality using BLEU.
5. Analyze unigram, bigram, trigram, and 4-gram precision.
6. Investigate the effect of preprocessing, particularly question-mark removal.
7. Compare the fine-tuned model with the original pretrained model.


## Project Workflow

```text
TED Talks Parallel Dataset
            │
            ▼
      Data Preprocessing
            │
            ├── Case Folding
            ├── Character Filtering
            ├── Duplicate Removal
            ├── Missing Value Handling
            └── Tokenization
            │
            ▼
      Train-Validation Split
            │
     ┌──────┼──────┬──────┬──────┐
     ▼      ▼      ▼      ▼      ▼
    70:30  75:25  80:20  85:15  90:10
     │      │      │      │      │
     └──────┴──────┴──────┴──────┘
            │
            ▼
      Pretrained MarianMT
            │
            ▼
        Fine-Tuning
            │
            ▼
        Model Evaluation
            │
            ├── BLEU
            ├── Unigram Precision
            ├── Bigram Precision
            ├── Trigram Precision
            └── 4-gram Precision
            │
            ▼
        Model Comparison
```
## Dataset

The project uses a parallel English–Indonesian dataset derived from TED Talks. The dataset contains aligned English and Indonesian sentences that allow the model to learn translation patterns between the two languages.
The data preparation process includes:
- Lowercasing text
- Removing irrelevant non-alphabetic characters
- Removing duplicate sentence pairs
- Handling missing values
- Tokenization
- Padding and truncation for model input
  
The research dataset contains 160,632 English–Indonesian sentence pairs used for training and validation. A separate test dataset was used to evaluate the final translation performance.

## Data Preprocessing
Preprocessing was performed before the data was passed to the Transformer model.
The main preprocessing steps were:
1. Case Folding : All text was converted to lowercase to create a more consistent representation of the input data.
2. Character Filtering : Characters considered irrelevant for the experiment were removed from the text. This preprocessing step was also investigated further through an experiment involving question marks.
3. Duplicate Removal : Duplicate sentence pairs were removed to reduce redundancy in the training data.
4. Missing Value Handling : Sentence pairs containing missing values were removed before model training.
5. Tokenization : The cleaned text was converted into model-compatible token representations using the MarianMT tokenizer.
6. Padding and Truncation : Input sequences were padded and truncated to maintain consistent sequence lengths during training.

## Model
This project uses MarianMT, a Transformer-based neural machine translation model. The pretrained checkpoint used in the original experiment was:

`Helsinki-NLP/opus-mt-en-id`

The model was then fine-tuned using the TED Talks English–Indonesian dataset.
Why MarianMT? MarianMT provides pretrained translation models that can be adapted to specific datasets through fine-tuning. Instead of training a translation model from scratch, the project starts from a pretrained English-to-Indonesian MarianMT model and adapts it to the characteristics of formal TED Talks data.

## Fine-Tuning Configuration
The main hyperparameters used in the original experiment were:
| Parameter | Value |
|---|---:|
| Model |	`Helsinki-NLP/opus-mt-en-id `|
| Learning Rate |	2 × 10⁻⁵ |
| Batch Size | 64 |
| Dropout	| 0.3 |
| Optimizer	| Adam |
| Epochs	| 3 |
| Early Stopping |	Applied |
| Tokenizer	| SentencePiece / MarianMT Tokenizer |
| Maximum Sequence Length	| 128 |

## Train-Validation Split

Five train-validation configurations were evaluated:
| Configuration | Training | Validation | 
|---|---:|---:|
|
