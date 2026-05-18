# USP at AmericasNLP 2026 — Culturally-Aware Image Captioning for Indigenous Languages

**USP team submission for the [AmericasNLP 2026 Shared Task](https://github.com/AmericasNLP/americasnlp2026) on Culturally Relevant Image Captioning for Indigenous Languages.**

We propose a two-stage cascade: **Qwen3-VL-8B-Instruct** generates culturally-prompted Spanish captions (zero-shot), which are then translated into each target language by a **fine-tuned NLLB-200-distilled-600M** model.

**Results:** 3rd place in Guaraní human evaluation (2.41/5.0), 5th in Bribri (1.09/5.0) among 8 teams.

---

## Table of Contents

1. [Task Overview](#task-overview)
2. [Languages](#languages)
3. [System](#system)
4. [Data Sources](#data-sources)
5. [Key Findings](#key-findings)
6. [Notebooks](#notebooks)
7. [Team](#team)
8. [References](#references)

---

## Task Overview

The **AmericasNLP 2026 Shared Task on Culturally Relevant Image Captioning** provides images from five Indigenous communities and asks systems to generate culturally grounded captions in each community's language. Unlike standard captioning benchmarks, correct answers require recognizing community-specific objects, practices, and symbols — not just literal visual content.

---

## Languages

| Language       | ISO Code   | Region              | Fine-tuned | Human eval rank |
|----------------|------------|---------------------|------------|-----------------|
| Guaraní        | `grn_Latn` | Paraguay, Brazil    | Yes        | 3rd / 5         |
| Maya Yucateco  | `yua_Latn` | Mexico (Yucatán)    | Yes        | —               |
| Nahuatl        | `nah_Latn` | Mexico (Veracruz)   | Yes        | —               |
| Wixárika       | `hch_Latn` | Mexico              | Yes        | —               |
| Bribri         | `bzd_Latn` | Costa Rica          | Yes        | 5th / 5         |

> **Note:** `bzd_Latn` (Bribri) and `yua_Latn` (Maya Yucateco) are **absent from the base NLLB-200 vocabulary** — `tokenizer.convert_tokens_to_ids("bzd_Latn")` returns id 3 (`<s>`), causing the base model to silently output English with no error. Both tokens are added as new special tokens before fine-tuning.

---

## System

```
Input Image
    │
    ▼
Qwen3-VL-8B-Instruct
+ language-specific cultural prompt (Spanish, zero-shot)
    │
    ▼
Intermediate Spanish Caption
    │
    ▼
Fine-tuned NLLB-200-distilled-600M
(one model per target language)
    │
    ▼
Indigenous Language Caption
```

**Stage 1 — Visual Captioning:**
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Each language has a dedicated Spanish-language cultural prompt listing community-specific objects, foods, mythology, and practices
- Inference: `do_sample=False`, `max_new_tokens=128`, `repetition_penalty=1.3`, images resized to 384×384 px
- Platform: Kaggle (NVIDIA T4)

**Stage 2 — Neural Machine Translation:**
- Base model: `facebook/nllb-200-distilled-600M`
- Fine-tuned separately per language on parallel corpora (see Data Sources)
- Beam search k=4, `max_length=256`
- Bribri uses `repetition_penalty=2.5`, `no_repeat_ngram_size=3` to suppress degenerate decoding
- Platform: Google Colab Pro (NVIDIA A100) for Wixárika/Nahuatl/Bribri/Guaraní; Kaggle (T4) for Maya

---

## Data Sources

| Language      | AmericasNLP 2023 | Extra corpus | Total pairs |
|---------------|-----------------|--------------|-------------|
| Guaraní       | ~14k            | Jojajovai    | ~46k        |
| Nahuatl       | ~16k            | Axolotl      | ~36k        |
| Wixárika      | ~9k             | Pywirrarika  | ~16k        |
| Bribri        | ~7.5k           | Feldman et al. | ~9.2k     |
| Maya Yucateco | —               | Iikim Translator + dev upsampling (20×) | ~11.8k |

All external data used in accordance with the AmericasNLP 2026 rules permitting *"any additional resources (external data, pretrained models, etc.)"*.

| Dataset | Source | Citation |
|---------|--------|----------|
| Jojajovai (Guaraní) | [pln-fing-udelar/jojajovai](https://github.com/pln-fing-udelar/jojajovai) | Chiruzzo et al., LREC 2022 |
| Pywirrarika (Wixárika) | [pywirrarika/wixarika-spanish](https://github.com/pywirrarika/wixarika-spanish) | — |
| Axolotl (Nahuatl) | [hackathon-pln-es/Axolotl-Spanish-Nahuatl](https://huggingface.co/datasets/hackathon-pln-es/Axolotl-Spanish-Nahuatl) | Gutierrez-Vasques et al., LREC 2016 |
| Feldman Corpus (Bribri) | [rolandocoto/bribri-coling2020](https://github.com/rolandocoto/bribri-coling2020) | Feldman & Coto-Solano, LREC 2020 |
| Iikim Translator (Maya) | Iikim Translator project | — |

---

## Key Findings

1. **NLLB-200 silently lacks vocabulary for Bribri and Maya.** `tokenizer.convert_tokens_to_ids("bzd_Latn")` returns id 3 (`<s>`), not an error. The base model outputs fluent English instead of the target language. Both tokens must be added as new special tokens before fine-tuning.

2. **Fine-tuning can hurt when domain mismatch is high.** For Guaraní, the base NLLB model (19.49 ChrF++) outperformed the fine-tuned version (17.57 ChrF++) on the dev set. The fine-tuned model was submitted based on expected better cultural coverage — confirmed by 3rd-place human evaluation. Domain mismatch between general MT training corpora and image captions is the primary bottleneck.

3. **Cultural prompting improves human-judged quality in ways ChrF++ does not capture.** Our 3rd-place Guaraní result was achieved without fine-tuning the VLM — purely through language-specific cultural prompts. See the paper for full analysis.

---

## Notebooks

### [`notebook1_captioning_and_translation.ipynb`](notebook1_captioning_and_translation.ipynb)
> **Platform:** Kaggle (NVIDIA T4)

Covers the full pipeline for all five languages:
- Stage 1: VLM captioning with cultural prompts (all languages)
- Stage 2: Translation with fine-tuned NLLB models (all languages)
- Guaraní dev set evaluation (base vs fine-tuned comparison)
- Maya fine-tuning (Kaggle, Iikim Translator corpus)
- Post-processing: loop detection and correction (Wixárika, Guaraní)
- Final submission assembly and verification

### [`notebook2_nmt_finetuning.ipynb`](notebook2_nmt_finetuning.ipynb)
> **Platform:** Google Colab Pro (NVIDIA A100)

Fine-tuning NLLB-200 for Wixárika, Nahuatl, Bribri, and Guaraní:
- NLLB vocabulary verification (identifies missing tokens)
- Data loaders for all corpora (AmericasNLP 2023 + extra)
- Training loop with Adafactor, gradient checkpointing
- Model backup to Google Drive

---

## Team

| Name | Institution | Contact |
|------|-------------|---------|
| Rafael M. Fernandes | Universidade de São Paulo (USP) | rafael.macario@usp.br |

---

## References

- Costa-jussà, M. R., et al. (2022). **No Language Left Behind: Scaling Human-Centered Machine Translation.** arXiv:2207.04672.
- Chiruzzo, L., et al. (2022). **Jojajovai: A Parallel Guaraní-Spanish Corpus for MT Benchmarking.** LREC 2022.
- Gutierrez-Vasques, X., Sierra, G., & Pompa, I. (2016). **Axolotl: A Large Corpus of Spanish-Nahuatl Parallel Text.** LREC 2016.
- Feldman, I., & Coto-Solano, R. (2020). **A Pipeline for Spoken Language Documentation of Bribri.** LREC 2020.
- Qwen Team (2025). **Qwen3-VL Technical Report.** arXiv:2511.21631.

---

*Part of the [AmericasNLP 2026](http://americasnlp.github.io/) evaluation campaign.*
