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
| Scenario 1 | 70% | 30% |
| Scenario 2 | 75% | 25% |
| Scenario 3 | 80% | 20% |
| Scenario 4 | 85% | 15% |
| Scenario 5 | 90% | 10% |

The test data was kept separate from the training and validation data. The 90:10 configuration produced the highest documented BLEU score in the experiment.

## Results
1. Model Comparison

The pretrained MarianMT model was evaluated before fine-tuning and compared with models fine-tuned using different train-validation split configurations.

| Model | Train : Validation | Unigram | Bigram | Trigram | 4-gram | BLEU |
|---|---:|---:|---:|---:|---:|---:|
| MarianMT Baseline | — | 0.523985 | 0.277398 | 0.156654 | 0.910766 | 21.30 |
| Fine-tuned MarianMT | 70:30 | 0.618372 | 0.387461 | 0.252331 | 0.167867 | 31.73 |
| Fine-tuned MarianMT | 75:25 | 0.619293 | 0.388465 | 0.253363 | 0.169127 | 31.86 |
| Fine-tuned MarianMT | 80:20 | 0.619970 | 0.389378 | 0.253643 | 0.168683 | 31.87 |
| Fine-tuned MarianMT | 85:15 | 0.620015 | 0.388849 | 0.252751 | 0.167850 | 31.80 |
| Fine-tuned MarianMT | 90:10 | 0.621940 | 0.391832 | 0.256202 | 0.171001 | 32.14 |

The baseline MarianMT model achieved a BLEU score of 21.30, while the fine-tuned configurations achieved BLEU scores above 31. The highest documented BLEU score was 32.14, obtained using the 90:10 train-validation configuration. The corresponding n-gram precision values were:
- Unigram: 0.621940
- Bigram: 0.391832
- Trigram: 0.256202
- 4-gram: 0.171001

2. Training History

The recorded training history for the 90:10 configuration is shown below.

| Epoch | Training Loss | Validation Loss | BLEU | Gen Len |
|---|---:|---:|---:|---:|
| 1 | 1.794600 | 1.667744 | 30.934200 | 16.858400 |
| 2 | 1.678300 | 1.642274 | 31.236900 | 16.874000 |
| 3 | 1.605700 | 1.635424 | 31.357800 | 16.876300 |

The training loss decreased consistently from 1.794600 in epoch 1 to 1.605700 in epoch 3. The validation loss also decreased from 1.667744 to 1.635424, while BLEU increased from 30.934200 to 31.357800 during the recorded training epochs. The original thesis describes the validation loss as relatively stable during the three recorded epochs.

3. Punctuation Preprocessing Experiment

An additional experiment investigated whether removing question marks during preprocessing affected translation performance.
| Preprocessing Configuration | Unigram | Bigram | Trigram | 4-gram | BLEU |
|---|---:|---:|---:|---:|---:|
| Question marks retained | 0.614268 | 0.382124	| 0.247077 | 0.162699 | 31.16 |
| Question marks removed | 0.621940	| 0.391832 | 0.256202 | 0.171001 | 32.14 |

## Technologies & Tools
Programming Language
- Python

NLP & Machine Learning
- Hugging Face Transformers
- MarianMT
- PyTorch
- SentencePiece

Evaluation
- SacreBLEU
- BLEU
- N-gram Precision

Data Processing
- Pandas
- NumPy

## Research Context

This project is based on my undergraduate thesis:

"MARIANMT BERBASIS TRANSFORMER UNTUK PENERJEMAHAN BAHASA INGGRIS-INDONESIA DENGAN FINE-TUNING DATA TECHNOLOGY, EDUCATION, AND DESIGN (TED) TALKS"

## Author

**Adinda Hermawan**  
