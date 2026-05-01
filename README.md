# AmericasNLP 2026 — USP Submission

**USP team submission for the [AmericasNLP 2026 Shared Task](http://americasnlp.github.io/) on Machine Translation for Indigenous Languages of the Americas.**

We fine-tune Meta's **NLLB-200** model for four low-resource indigenous languages: Guarani, Wixarika, Nahuatl, and Bribri.

---

## Table of Contents

1. [Task Overview](#task-overview)
2. [Languages](#languages)
3. [Model & Approach](#model--approach)
4. [Repository Structure](#repository-structure)
5. [Requirements](#requirements)
6. [Usage](#usage)
7. [Team](#team)
8. [References](#references)

---

## Task Overview

The **AmericasNLP 2026 Shared Task** focuses on machine translation between Spanish and indigenous languages of the Americas. These languages are severely under-resourced, presenting unique challenges for neural machine translation systems.

Our approach builds on Meta AI's **No Language Left Behind (NLLB-200)** multilingual translation model, which already provides coverage for several indigenous languages. We fine-tune this model on task-provided parallel corpora to maximise translation quality for the four target languages.

---

## Languages

| Language   | ISO Code | Language Family    | Direction        |
|------------|----------|--------------------|------------------|
| Guarani    | `gn`     | Tupian             | Spanish ↔ Guarani |
| Wixarika   | `hch`    | Uto-Aztecan        | Spanish ↔ Wixarika |
| Nahuatl    | `nah`    | Uto-Aztecan        | Spanish ↔ Nahuatl |
| Bribri     | `bzd`    | Chibchan           | Spanish ↔ Bribri |

---

## Model & Approach

- **Base model:** [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M) (or the 1.3 B / 3.3 B variant)
- **Fine-tuning strategy:** Supervised sequence-to-sequence fine-tuning on the AmericasNLP 2026 parallel corpora using the Hugging Face `transformers` and `datasets` libraries.
- **Training framework:** PyTorch + Hugging Face `Seq2SeqTrainer`
- **Evaluation metric:** chrF++ (primary), BLEU (secondary)

---

## Repository Structure

```
americasnlp-2026-USP/
├── data/                  # Preprocessed datasets (not tracked by git)
│   ├── train/
│   ├── dev/
│   └── test/
├── scripts/               # Training and evaluation scripts
│   ├── train.py           # Fine-tuning script
│   ├── evaluate.py        # Evaluation script
│   └── preprocess.py      # Data preprocessing utilities
├── configs/               # Experiment configuration files (YAML)
├── outputs/               # Model checkpoints and predictions (not tracked)
├── requirements.txt       # Python dependencies
└── README.md
```

> **Note:** `data/` and `outputs/` are excluded from version control. Download the official AmericasNLP 2026 data from the task organisers and place it under `data/`.

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA-capable GPU recommended (≥ 16 GB VRAM for the 600 M parameter model)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Purpose |
|---------|---------|
| `transformers` | NLLB-200 model & tokeniser |
| `datasets` | Dataset loading & processing |
| `sacrebleu` | BLEU / chrF++ evaluation |
| `sentencepiece` | Tokenisation |
| `accelerate` | Distributed / mixed-precision training |

---

## Usage

### 1. Preprocessing

```bash
python scripts/preprocess.py \
    --src_lang spa_Latn \
    --tgt_lang grn_Latn \
    --input_dir data/raw/guarani \
    --output_dir data/processed/guarani
```

### 2. Fine-tuning

```bash
python scripts/train.py \
    --model_name_or_path facebook/nllb-200-distilled-600M \
    --src_lang spa_Latn \
    --tgt_lang grn_Latn \
    --train_file data/processed/guarani/train.json \
    --validation_file data/processed/guarani/dev.json \
    --output_dir outputs/guarani \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5
```

Replace `grn_Latn` with the appropriate NLLB language code for each target language:

| Language | NLLB code |
|----------|-----------|
| Guarani  | `grn_Latn` |
| Wixarika | `hch_Latn` |
| Nahuatl  | `nhn_Latn` |
| Bribri   | `bzd_Latn` |

### 3. Evaluation

```bash
python scripts/evaluate.py \
    --model_dir outputs/guarani \
    --test_file data/processed/guarani/test.json \
    --src_lang spa_Latn \
    --tgt_lang grn_Latn
```

---

## Team

| Name | Institution | Contact |
|------|-------------|---------|
| Rafael Macario | Universidade de São Paulo (USP) | rafael.macario@usp.br |

---

## References

- Costa-jussà, M. R., et al. (2022). **No Language Left Behind: Scaling Human-Centered Machine Translation.** arXiv:2207.04672. [https://arxiv.org/abs/2207.04672](https://arxiv.org/abs/2207.04672)
- Mager, M., et al. (2021). **AmericasNLP 2021 Shared Task on Open-Domain Dialog Systems and Slot Filling.** In *Proceedings of the First Workshop on Natural Language Processing for Indigenous Languages of the Americas (AmericasNLP)*, ACL-IJCNLP.
- Hugging Face `transformers` library: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

---

*This project is part of the [AmericasNLP 2026](http://americasnlp.github.io/) evaluation campaign.*
