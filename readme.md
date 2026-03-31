# Epileptic Seizure Prediction Using Patient-Adaptive Transformer Networks

[![arXiv](https://img.shields.io/badge/arXiv-2603.26821-b31b1b.svg)](https://arxiv.org/abs/2603.26821)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

Official implementation of the paper:

> **Epileptic Seizure Prediction Using Patient-Adaptive Transformer Networks**  
> Mohamed MAHDI, Asma BAGHDADI  
> National Engineering School of Tunis, Tunis El Manar University · REGIM-Lab, University of Sfax  
> 📄 [arXiv:2603.26821](https://arxiv.org/abs/2603.26821)

---

## Overview

Epilepsy affects more than 50 million people worldwide. Because seizures occur without warning, reliable early prediction is a major clinical challenge. This work proposes a **patient-specific, short-horizon seizure forecasting framework** that combines:

1. **Self-supervised GPT-style pretraining** — the transformer learns general EEG temporal dynamics through autoregressive next-token prediction, without requiring seizure annotations.
2. **Patient-specific supervised fine-tuning** — the pretrained backbone is adapted per patient to predict whether a seizure will occur within the **next 30 seconds**.

By grounding the model in individualized neural patterns rather than population-level statistics, the framework directly addresses inter-patient variability — one of the core challenges in clinical EEG analysis.

---

## Method

### Problem Formulation

Given a 30-second multichannel EEG segment $X \in \mathbb{R}^{C \times T}$, the model outputs a binary prediction:

- **y = 1** → A seizure will occur within the next 30 seconds *(pre-ictal)*
- **y = 0** → No seizure will occur within this horizon *(inter-ictal / background)*

### Pipeline

```
EEG Signal
    │
    ▼
[1] Noise-Aware Preprocessing
    • PSD-based noise detection across channels
    • Adaptive notch filtering (no fixed 50/60 Hz assumption)
    │
    ▼
[2] EEG Tokenization
    • Z-score normalization → clip at ±5σ → map to [0,1]
    • Quantize to L=512 discrete amplitude levels
    • One token per sample → sequence Z = [z₁, z₂, ..., zT]
    │
    ▼
[3] Self-Supervised Pretraining (GPT-style)
    • Autoregressive next-token prediction
    • Dual loss: L_CE + 0.1 × L_MSE (waveform reconstruction)
    • 5,000 steps, patient-specific
    │
    ▼
[4] Patient-Specific Fine-Tuning
    • Transfer pretrained token & positional embeddings
    • Channel projection layer (C → 128 dims)
    • Global average pooling over time
    • Binary classification head: 128 → 2
    • Weighted cross-entropy for class imbalance
    • 5,000 steps, LR = 3×10⁻⁵
    │
    ▼
Binary Seizure Alarm (seizure within 30s: yes/no)
```

### Key Design Choices

- **Adaptive filtering** — rather than applying fixed notch filters, the pipeline detects high-power, low-variance frequency components per session, enabling session-specific noise removal without discarding physiologically relevant rhythms.
- **Raw waveform tokenization** — the model operates on quantized raw samples (no spectrograms, no handcrafted features), enabling end-to-end learning of both micro-scale waveform morphology and macro-scale temporal structure within large context windows.
- **Dual pretraining loss** — cross-entropy over tokens encourages accurate symbolic transitions; the auxiliary MSE term on dequantized predictions preserves physiologically meaningful waveform fidelity.
- **Patient-specific paradigm** — models are trained and evaluated independently per subject with no cross-patient data mixing, reflecting realistic clinical deployment conditions where seizure signatures vary substantially across individuals.

---

## Results

### Per-Patient Performance

| Patient | Val Acc | Precision | Recall | F1 | FAR (h⁻¹) | P-delay (s) | Sensitivity |
|---|---|---|---|---|---|---|---|
| aaaaaaac | **97.06%** | 1.0000 | 0.8333 | **0.9091** | 0.00 | 6.89 | 100% |
| aaaaabnn | **94.44%** | 0.8750 | 0.7778 | **0.8235** | 0.00 | 12.55 | 100% |
| aaaaadpj | **93.94%** | 0.7500 | 1.0000 | **0.8571** | 13.82 | 30.00 | 100% |

All three patients achieved **100% sensitivity** — every seizure was detected — with F1 scores between 0.82 and 0.91 and validation accuracies above 93%.

*Top: Binary alarms (red = seizure predicted, green = normal) with ground-truth seizure windows in orange. Bottom: Continuous seizure probability with 0.5 decision threshold.*

---

## Dataset

Experiments use recordings from the **TUH EEG Seizure Corpus** (Temple University Hospital).

| Patient | Sessions | Rec. Hours | Seizures | Segments | Pre-ictal % | Fs (Hz) | Channels |
|---|---|---|---|---|---|---|---|
| aaaaaaac (s001–s005) | 9 | ~0.83 h | 3 | 170 | ~35% | 250 | 33 |
| aaaaabnn (s001–s004) | 13 | ~1.28 h | 4 | 266 | ~14% | 250 | 32–128 |
| aaaaadpj (s003–s005) | 7 | ~0.79 h | 7 | 165 | ~13% | 250 | 32–41 |

- **Segmentation:** 30-second sliding windows — 50% overlap for training, 75% for inference
- **Channels:** Standardized per patient via zero-padding or truncation
- **Labels:** A segment is labeled pre-ictal if any annotated seizure onset falls within the following 30-second window

---

## Installation

**Requirements:** Python 3.11+, CUDA recommended (GTX 1650 or better)

```bash
git clone https://github.com/Mahdyy02/gpt-eeg-transformer.git
cd gpt-eeg-transformer

pip install torch torchvision torchaudio
pip install mne
pip install numpy pandas matplotlib scikit-learn
```

---

## Usage

### Step 1 — Preprocess raw EEG recordings

```bash
python filter_and_merge.py
python verify_filtered_edf.py
```

### Step 2 — Self-supervised pretraining

```bash
python train_eeg_transformer.py
```

Output: `eeg_transformer.pth`

### Step 3 — Evaluate pretraining (optional)

```bash
python inference_eeg_transformer.py \
  --num_tokens 1500 \
  --temperature 0.8 \
  --top_k 100
```

Generates synthetic EEG to verify that the pretrained model has learned meaningful signal structure.

### Step 4 — Patient-specific fine-tuning

```bash
python train_seizure_prediction.py
```

Output: `seizure_prediction_finetuned.pth`

### Step 5 — Seizure prediction on new recordings

```bash
python inference_seizure_prediction.py \
  --checkpoint seizure_prediction_finetuned.pth \
  --edf "aaaaadpj/s005_2006/03_tcp_ar_a/aaaaadpj_s005_t000.edf" \
  --csv "aaaaadpj/s005_2006/03_tcp_ar_a/aaaaadpj_s005_t000.csv" \
  --output predictions.png
```

### Visualization utilities

```bash
python plot.py                    # General EEG signal plots
python plot_all_channels.py       # Multi-channel overview
python plot_filtered_signals.py   # Raw vs. filtered comparison
python plot_timefreq.py           # Time-frequency spectrograms
python compute_patient_stats.py   # Per-patient statistics
python read_edf.py                # Inspect raw EDF files
```

---

## Project Structure

```
gpt-eeg-transformer/
├── train_eeg_transformer.py          # Stage 1: self-supervised GPT pretraining
├── inference_eeg_transformer.py      # Synthetic EEG generation from pretrained model
├── train_seizure_prediction.py       # Stage 2: patient-specific fine-tuning
├── inference_seizure_prediction.py   # Seizure prediction on new EDF recordings
├── filter_and_merge.py               # Adaptive noise-aware preprocessing
├── compute_patient_stats.py          # Per-patient dataset statistics
├── read_edf.py                       # EDF file reader
├── verify_filtered_edf.py            # Validate filtered EDF output
├── plot.py                           # General signal visualization
├── plot_all_channels.py              # Multi-channel signal plots
├── plot_filtered_signals.py          # Raw vs. filtered signal comparison
├── plot_timefreq.py                  # Time-frequency analysis
├── architecture.svg                  # Model architecture diagram
├── predictions.png                   # Example prediction timeline output
└── presentation_complete.html        # Full project presentation
```

---

## Hyperparameters

### Tokenization & Pretraining

| Hyperparameter | Value |
|---|---|
| Quantization levels (L) | 512 |
| Block size (context window) | 512 samples |
| Normalization | Z-score |
| Clipping factor (k) | 5 |
| Tokens per sample | 1 |
| Embedding dimension | 128 |
| Transformer layers | 4 |
| Attention heads | 4 |
| Positional embeddings | Learned |
| Gradient accumulation | 8 steps |
| MSE loss weight (λ) | 0.1 |
| Pretraining steps | 5,000 |

### Fine-Tuning

| Hyperparameter | Value |
|---|---|
| Segment duration | 30 s |
| Prediction horizon | 30 s |
| Sampling rate | 250 Hz |
| Sequence length | 7,500 samples |
| Batch size | 16 |
| Gradient accumulation | 4 steps |
| Learning rate | 3×10⁻⁵ |
| Dropout | 0.2 |
| Loss function | Weighted Cross-Entropy |
| Mixed precision | Enabled (FP16) |
| Fine-tuning steps | 5,000 |

---

## Limitations & Future Work

**Current limitations:**
- Proof-of-concept on three patients — large-scale generalization is untested
- Short recording durations limit pre-ictal segment availability
- Class imbalance (~13–35% positive class depending on patient)
- False alarm rates vary across subjects; decision threshold not yet tuned

**Planned extensions:**
- Large-scale validation across the full TUH EEG corpus
- Sensitivity–false alarm trade-off optimization (threshold tuning, temporal aggregation)
- Multi-patient and cross-patient generalization studies
- Real-time streaming inference for wearable or bedside deployment
- Multi-class seizure type classification (fnsz, gnsz, cpsz, etc.)
- Attention visualization to identify pre-ictal EEG signatures

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{mahdi2026seizure,
  title   = {Epileptic Seizure Prediction Using Patient-Adaptive Transformer Networks},
  author  = {Mahdi, Mohamed and Baghdadi, Asma},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.26821}
}
```

---

## References

1. **TUH EEG Seizure Corpus** — Harati et al., Temple University Hospital EEG Database.
2. **Attention Is All You Need** — Vaswani et al., NeurIPS 2017.
3. **A Transformer-based Framework for Multivariate Time Series Representation Learning** — Zerveas et al., KDD 2021.
4. **MNE-Python** — Gramfort et al., 2013.
5. **Paper** — [arXiv:2603.26821](https://arxiv.org/abs/2603.26821)

---

**Project status:** ✅ Functional | 🚧 Research in Progress  
**Live presentation:** [mahdyy02.github.io/gpt-eeg-transformer](https://mahdyy02.github.io/gpt-eeg-transformer/presentation_complete.html)
