# AmericasNLP 2026 — USP Submission

**USP team submission for the [AmericasNLP 2026 Shared Task](http://americasnlp.github.io/) on Machine Translation for Indigenous Languages of the Americas.**

We fine-tune Meta's **NLLB-200** model for four low-resource indigenous languages of the Americas: **Guarani, Wixarika, Nahuatl, and Bribri**.

---

## Table of Contents

1. [Task Overview](#task-overview)
2. [Languages](#languages)
3. [Model & Approach](#model--approach)
4. [Notebooks](#notebooks)
5. [Team](#team)
6. [References](#references)

---

## Task Overview

The **AmericasNLP 2026 Shared Task** focuses on machine translation between Spanish and indigenous languages of the Americas. These languages are severely under-resourced, presenting unique challenges for neural machine translation systems.

Our approach builds on Meta AI's **No Language Left Behind (NLLB-200)** multilingual translation model, which already provides coverage for several indigenous languages. We fine-tune this model on the task-provided parallel corpora to maximise translation quality for the four target languages.

---

## Languages

| Language   | ISO Code | Language Family | Translation Direction |
|------------|----------|-----------------|-----------------------|
| Guarani    | `gn`     | Tupian          | Spanish ↔ Guarani     |
| Wixarika   | `hch`    | Uto-Aztecan     | Spanish ↔ Wixarika    |
| Nahuatl    | `nah`    | Uto-Aztecan     | Spanish ↔ Nahuatl     |
| Bribri     | `bzd`    | Chibchan        | Spanish ↔ Bribri      |

---

## Model & Approach

- **Base model:** [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
- **Fine-tuning strategy:** Supervised sequence-to-sequence fine-tuning on the AmericasNLP 2026 parallel corpora using the Hugging Face `transformers` and `datasets` libraries.
- **Training framework:** PyTorch + Hugging Face `Seq2SeqTrainer` (Google Colab)
- **Evaluation metric:** chrF++ (primary), BLEU (secondary)

---

## Notebooks

This repository contains two Jupyter notebooks that cover the full pipeline:

### 1. Fine-tuning — [`FINE_TUNING_guarani,_wixarika,_nahuatl,_bribri-2.ipynb`](FINE_TUNING_guarani,_wixarika,_nahuatl,_bribri-2.ipynb)

> **Environment:** Google Colab (GPU required)

**What it does:**
- Checks GPU availability and mounts Google Drive to read pre-processed data
- Loads the NLLB-200 model and tokeniser
- Fine-tunes the model on the AmericasNLP 2026 parallel corpora for all four target languages
- Saves model checkpoints back to Drive

**Key libraries:** `transformers`, `datasets`, `sentencepiece`, `torch`

---

### 2. Evaluation & Inference — [`notebookb2d7b39cba-4.ipynb`](notebookb2d7b39cba-4.ipynb)

> **Environment:** Kaggle (GPU recommended)

**What it does:**
- Clones the [AmericasNLP 2026 repository](https://github.com/AmericasNLP/americasnlp2026) to access test sets
- Downloads previously generated captions / model outputs from Drive
- Loads the fine-tuned NLLB-200 checkpoint
- Runs inference on the test split for each language
- Computes chrF++ and BLEU scores using `sacrebleu`

**Key libraries:** `transformers`, `sentencepiece`, `sacrebleu`, `gdown`, `torch`

---

## Team

| Name | Institution | Contact |
|------|-------------|---------|
| Rafael Macario | Universidade de São Paulo (USP) | rafael.macario@usp.br |

---

## References

- Costa-jussà, M. R., et al. (2022). **No Language Left Behind: Scaling Human-Centered Machine Translation.** arXiv:2207.04672. <https://arxiv.org/abs/2207.04672>
- Mager, M., et al. (2021). **Findings of the AmericasNLP 2021 Shared Task.** In *Proceedings of the First Workshop on Natural Language Processing for Indigenous Languages of the Americas (AmericasNLP)*, ACL-IJCNLP.
- Hugging Face `transformers`: <https://github.com/huggingface/transformers>

---

*This project is part of the [AmericasNLP 2026](http://americasnlp.github.io/) evaluation campaign.*
