# 🔍 TRIA — Hate Speech Detection in Arabic & Multilingual Context

> **TRIA** (likely referring to a project/research initiative) focuses on fine-tuning pre-trained transformer models to detect hate speech, with a primary focus on **Arabic** (social media, MSA, dialects) and a secondary **multilingual** component (English, French, Arabic).

---

## 📂 Project Structure

```
TRIA/
├── README.md                                      # This file
├── qarib-arabic-hate-speech.ipynb               # QARiB model fine-tuning & evaluation
├── arabert-hate-speech-v32d9d8f58ec.ipynb        # AraBERT fine-tuning & comparison
├── bert-robust-hate-speech-last-attempt.ipynb   # Robust multilingual BERT (mBERT) attempt
├── link to the presentation.txt                 # Canva presentation link
└── Fine-Tuning Pre-trained transofrmer models for hate speech detection.pdf  # Reference document
```

---

## 🧠 Project Overview

### Goal
Fine-tune pre-trained transformer models for **binary hate speech classification** (`0 = not hate`, `1 = hate`) and compare performance **before vs after fine-tuning**, applying robust anti-overfitting practices.

### Key Languages & Domains
- **Arabic** — Modern Standard Arabic (MSA), social media tweets, Tunisian / Egyptian dialects (QARiB, AraBERT notebooks)
- **Multilingual** — English, French, Arabic dialects + Classical Arabic (mBERT notebook)

---

## 📘 Notebooks

### 1. `qarib-arabic-hate-speech.ipynb`
**Model:** `QARiB`  
**What it is:** Pre-trained on **420M Arabic tweets + MSA**, specifically designed for social media Arabic.

| Parameter | Value | Why |
|---|---|---|
| `MAX_LEN` | 80 | Covers 95%+ of Arabic social media posts; faster than 128 |
| `BATCH_SIZE` | 64 / GPU | Safe in 16 GB T4 at seq=80; 2× throughput vs 32 |
| `GRAD_ACCUM` | 2 | Effective batch = 64×2×2 GPUs = **256** |
| `EPOCHS` | 3 + early stop | Sufficient for BERT fine-tuning |
| `LR` | `3e-5` | Scaled for effective batch 256 |
| `LABEL_SMOOTH` | 0.1 | Regularisation, measurably improves F1 |
| LLRD decay | 0.9 | Per-layer LR — lower layers trained more conservatively |

**Notes:**
- No pip installs — uses Kaggle pre-installed packages only.
- Works with any `transformers` version (`4.x → 5.x`).
- **Before vs After Fine-Tuning Comparison** is the core analysis.

---

### 2. `arabert-hate-speech-v32d9d8f58ec.ipynb`
**Model:** `AraBERT` (Arabic BERT)  
**Focus:** Before vs After fine-tuning comparison with speed/accuracy trade-offs.

| Setting | Value | Purpose |
|---|---|---|
| `MAX_LEN` | 64 | 4× faster attention vs 128 |
| `BATCH_SIZE` | 64 / GPU | 2× GPU utilisation |
| `GRAD_ACCUM` | 2 | Effective batch = **256** |
| `EPOCHS` | 3 + early stop | Cuts wasted epochs |
| `LR` | `3e-5` | Scaled for large effective batch |
| `LABEL_SMOOTH` | 0.1 | Regularisation, better F1 |
| LLRD | decay = 0.9 | Per-layer LR, better accuracy |

**Target:** ~2h training on T4×2 GPUs with higher accuracy than baseline AraBERT.

---

### 3. `bert-robust-hate-speech-last-attempt.ipynb`
**Model:** `bert-base-multilingual-cased` (`mBERT`)  
**Task:** Binary hate speech (`0` / `1`) across multiple languages.

**Languages Covered:**
- English
- French
- Arabic (Tunisian / Egyptian dialects + Classical)

**Anti-Overfitting Measures (Robust Design):**

| Problem | Fix Applied |
|---|---|
| Suspicious lexical features (profanity flags, keyword lists) | **Removed** — only raw text is used |
| Model memorising training set | Dropout `0.3` on classifier head + weight decay `0.05` |
| Learning too fast | Lower LR (`1e-5`) + warmup `15%` |
| No early stopping | Val F1 patience = 2 — saves best checkpoint and halts |
| Data leakage | Split **before** any feature engineering; stratified by label |
| Class-trivial patterns | Minimal text cleaning — no hate-keyword stripping |

**Pipeline:**  
`0. Setup → 1. EDA → 2. Leak Audit → 3. Preprocessing → 4. Fine-Tuning → 5. Evaluation`

---

## ⚙️ Common Environment & Setup

- **Frameworks:** `transformers` (`5.0.0`), `torch` (`2.10.0`)
- **Hardware (target):** Kaggle T4 GPUs (`×2`, DataParallel)
- **No external pip installs** required — relies on Kaggle pre-installed packages.
- **Compatible:** Any `transformers` version from `4.x` to `5.x`.

---

## 🛡️ Anti-Leakage & Robustness Strategy

This project explicitly addresses common pitfalls in hate-speech detection:

1. **Leak Audit** — Check for data leakage before any feature engineering.
2. **Stratified Split** — Performed **before** preprocessing or feature extraction.
3. **No Lexical Shortcuts** — Removed profanity flags / keyword lists to prevent the model from learning superficial patterns.
4. **Minimal Cleaning** — No stripping of hate keywords, so the model learns real linguistic patterns rather than cleaning artifacts.
5. **Regularisation Stack:**
   - Label smoothing (`0.1`)
   - Layer-wise Learning Rate Decay (`LLRD` decay `0.9`)
   - Weight decay (`0.05` for mBERT)
   - Dropout (`0.3` on head)
6. **Early Stopping** — Based on validation F1 (`patience = 2`).

---

## 📊 Before vs After Fine-Tuning

All three notebooks follow a **before / after comparison** structure:

1. **Baseline (pre-trained)** — Evaluate the raw model on the hate-speech task.
2. **Fine-Tuned** — Train with the hyperparameters above.
3. **Compare Metrics** — Accuracy, F1, and other classification metrics.
4. **Report Improvements** — Measure how fine-tuning improves detection over the pre-trained baseline.

> **Note:** The exact numeric results (F1 scores, accuracy, confusion matrices) are contained within each notebook's evaluation sections (`5. Evaluation` for mBERT, final comparison cells for QARiB / AraBERT).

---

## 📎 Additional Resources

- **Presentation:** [Canva Link](https://canva.link/jq8xmioy33s7lha) — `link to the presentation.txt`
- **Reference PDF:** `Fine-Tuning Pre-trained transofrmer models for hate speech detection.pdf`

---

## ✅ Key Takeaways

- **QARiB** is best for **social media Arabic** (tweets, informal text) due to its 420M Arabic tweet pre-training.
- **AraBERT** provides a strong **Arabic BERT** baseline with faster attention (`MAX_LEN = 64`).
- **mBERT** extends the approach to **multilingual** settings (English, French, Arabic dialects + Classical) with a robust anti-overfitting pipeline.
- **Anti-leak practices** (stratified split before engineering, removal of lexical shortcuts, label smoothing, LLRD, dropout, weight decay, early stopping) are central to this project's methodology.
- **No external dependencies** — everything runs on Kaggle's pre-installed stack.

---

*Created for the TRIA project — analysis of fine-tuning pre-trained transformers for hate speech detection.*
