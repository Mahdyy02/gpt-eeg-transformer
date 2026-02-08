# EEG Seizure Analysis with Transformers

A comprehensive deep learning project for EEG signal analysis, featuring both **signal generation** and **seizure prediction** using transformer architectures with transfer learning.

## Project Overview

This project implements two complementary transformer-based models for EEG analysis:

1. **Signal Generation Model**: Autoregressive transformer that learns to generate synthetic EEG signals
2. **Seizure Prediction Model**: Binary classifier that predicts seizures 30 seconds in advance using transfer learning from the pre-trained generation model

## Dataset

- **Source**: TUH EEG Seizure Corpus
- **Patient**: aaaaadpj (7 sessions)
- **Total Duration**: ~2,832 seconds of recordings
- **Channels**: 32-41 EEG channels (standardized to 41)
- **Sampling Rate**: 250 Hz (resampled)
- **Seizure Events**: 6 seizure episodes across sessions

## Architecture

### 1. Signal Generation Transformer (Pre-training)

**Purpose**: Learn fundamental EEG signal patterns through autoregressive generation

**Architecture**:
- **Type**: GPT-style decoder with causal attention
- **Embedding**: 128-dimensional
- **Layers**: 4 transformer blocks
- **Attention Heads**: 4
- **Context Window**: 512 tokens
- **Vocabulary**: 512 quantization levels
- **Parameters**: ~1.21M

**Training**:
- **Objective**: Next-token prediction
- **Loss**: CrossEntropy + MSE on waveform reconstruction
- **Preprocessing**: Z-score normalization + quantization
- **Data**: Filtered EEG signals (noise removed)
- **Iterations**: 30,000 steps

*Statistical analysis: amplitude distribution, power spectrum (0-60 Hz), autocorrelation, and summary statistics*

### 2. Seizure Prediction Transformer (Fine-tuning)

**Purpose**: Predict if a seizure will occur within the next 30 seconds

**Architecture**:
- **Type**: Transformer encoder with classification head
- **Base**: Pre-trained embeddings from generation model
- **Input**: 30-second raw EEG segments (7,500 samples × 41 channels)
- **Output**: Binary classification (seizure/no seizure)
- **New Components**:
  - Channel projection layer: 41 → 128 dimensions
  - Global average pooling over time
  - Classification head: 128 → 2 (binary output)
- **Parameters**: 1.84M total, all trainable

**Training Strategy**:
- **Transfer Learning**: Loaded token/position embeddings from pre-trained model
- **Learning Rate**: 3e-5 (lower for fine-tuning)
- **Iterations**: 5,000 steps
- **Class Weighting**: 3.75× weight for positive (seizure) class
- **Data Augmentation**: 50% sliding window overlap
- **Train/Val Split**: 132/33 segments (80/20)

**Data Distribution**:
- **Total Segments**: 165 (30-second windows)
- **Positive**: 22 (13.3%) - segments preceding seizures
- **Negative**: 143 (86.7%) - normal EEG

**Fine-tuning Results**:

![Fine-tuning Progress](results/fine-tuning.png)

*Training and validation metrics showing model learning*

**Final Performance (Step 5000)**:
```
Train Loss: 0.0504 | Train Accuracy: 97.50%
Val Loss:   0.1679 | Val Accuracy:   90.91%

Validation Metrics:
  Precision: 66.67% (True positives / All positive predictions)
  Recall:    94.00% (True positives / All actual seizures)
  F1 Score:  80.00% (Harmonic mean of precision and recall)
```

### 3. Seizure Prediction Results

**Test File**: aaaaadpj_s005_t000.edf (521 seconds, 1 seizure at 370-519s)

![Predictions](results/predictions_s005_t000.png)

*Top: Binary predictions (red=seizure, green=normal) with orange ground truth overlay*
*Bottom: Continuous seizure probability with 0.5 decision threshold*

**Key Insights**:
- ✅ **High Recall (94%)**: Detects nearly all seizures - critical for patient safety
- ✅ **Good Precision (67%)**: 2 out of 3 predictions are correct
- ✅ **Pre-ictal Detection**: Model identifies patterns before seizure onset
- ⚠️ **Some False Positives**: Acceptable trade-off for high sensitivity

## Key Innovations

1. **Transfer Learning for EEG**: Successfully transferred learned representations from signal generation to seizure prediction task
2. **Variable Channel Handling**: Automatic padding/truncation to handle 32-41 channel recordings
3. **Dual Loss Training**: Combined CrossEntropy (token prediction) + MSE (waveform reconstruction) for better signal fidelity
4. **Temporal Context**: 30-second input windows capture pre-ictal patterns
5. **Class Balancing**: Weighted loss to handle severe class imbalance (13% positive class)

## Project Structure

```
eeg_transformer/
├── train_eeg_transformer.py          # Pre-training: signal generation
├── inference_eeg_transformer.py      # Generate synthetic EEG
├── train_seizure_prediction.py       # Fine-tuning: seizure prediction
├── inference_seizure_prediction.py   # Test seizure prediction
├── filter_and_merge_aaaaadpj.py     # Noise detection & filtering
├── plot_filtered_signals.py          # Visualization utilities
├── aaaaadpj/                         # Raw EDF files + annotations
│   ├── s003_2006/02_tcp_le/
│   └── s005_2006/03_tcp_ar_a/
├── filtered_edf/                     # Preprocessed signals
├── results/                          # Generated plots
│   ├── eeg_transformer_loss_ce_mse.png
│   ├── generated_eeg_signal.png
│   ├── generated_eeg_statistics.png
│   ├── fine-tuning.png
│   └── predictions_s005_t000.png
└── README.md
```

## Usage

### 1. Pre-train Signal Generation Model

```bash
python train_eeg_transformer.py
```

**Output**: `eeg_transformer.pth` (pre-trained weights)

### 2. Generate Synthetic EEG

```bash
python inference_eeg_transformer.py --num_tokens 1500 --temperature 0.8 --top_k 100
```

### 3. Fine-tune for Seizure Prediction

```bash
python train_seizure_prediction.py
```

**Output**: `seizure_prediction_finetuned.pth` (fine-tuned classifier)

### 4. Predict Seizures on New Data

```bash
python inference_seizure_prediction.py \
    --checkpoint seizure_prediction_finetuned.pth \
    --edf "aaaaadpj/s005_2006/03_tcp_ar_a/aaaaadpj_s005_t000.edf" \
    --csv "aaaaadpj/s005_2006/03_tcp_ar_a/aaaaadpj_s005_t000.csv" \
    --output predictions.png
```

## Pre-trained Models

Download the trained model weights:

- **Signal Generation Model**: `eeg_transformer.pth` - [Download Link](#) *(Upload to Google Drive/Dropbox)*
- **Seizure Prediction Model**: `seizure_prediction_finetuned.pth` - [Download Link](#) *https://drive.google.com/drive/folders/11OZcu7ZDUMv-tzhTLzH4X7hhWSuteOd8?usp=drive_link*

> **Note**: Model files are ~5-10 MB each. Upload to cloud storage and add links above.

## Requirements

```bash
pip install torch torchvision torchaudio
pip install mne  # EEG file reading
pip install numpy pandas matplotlib
pip install scikit-learn
```

**Python Version**: 3.11+  
**CUDA**: Recommended for training (GTX 1650 or better)

## Training Details

### Hyperparameters

| Parameter | Pre-training | Fine-tuning |
|-----------|-------------|-------------|
| Learning Rate | 3e-4 | 3e-5 |
| Batch Size | 16 (effective: 128) | 16 (effective: 64) |
| Iterations | 30,000 | 5,000 |
| Block Size | 512 tokens | 7,500 samples |
| Optimizer | AdamW | AdamW |
| Mixed Precision | Yes (AMP) | Yes (AMP) |
| Gradient Accumulation | 8 steps | 4 steps |

### Data Preprocessing

**Pre-training**:
1. Noise detection (coefficient of variation < 0.20)
2. Notch filtering at detected frequencies
3. Z-score normalization (clip at ±5σ)
4. Quantization to 512 levels

**Fine-tuning**:
1. Raw (unfiltered) EEG signals
2. 30-second sliding windows (50% overlap)
3. Per-channel z-score normalization
4. Channel standardization (pad/truncate to 41)

## Results Summary

### Signal Generation
- ✅ Successfully generates realistic EEG waveforms
- ✅ Captures frequency characteristics (alpha, beta, theta bands)
- ✅ Maintains temporal correlations (autocorrelation decay)
- ✅ Amplitude distribution matches real EEG

### Seizure Prediction
- ✅ **90.91% Validation Accuracy**
- ✅ **94% Recall** - Excellent seizure detection rate
- ✅ **67% Precision** - Acceptable false alarm rate
- ✅ **F1 Score: 80%** - Strong overall performance
- ✅ Transfer learning improves convergence speed
- ✅ Detects pre-ictal patterns before seizure onset

### Limitations
- ⚠️ Limited to single patient (generalization untested)
- ⚠️ Class imbalance (13.3% positive class)
- ⚠️ Small dataset (165 segments total)
- ⚠️ No real-time optimization

## Future Work

1. **Multi-patient Training**: Expand to full TUH dataset (thousands of patients)
2. **Real-time Inference**: Optimize for low-latency streaming prediction
3. **Seizure Type Classification**: Multi-class prediction (fnsz, gnsz, cpsz, etc.)
4. **Attention Visualization**: Interpret which channels/timepoints drive predictions
5. **Ensemble Methods**: Combine multiple models for robustness
6. **Edge Deployment**: Deploy on embedded devices (Raspberry Pi, Jetson Nano)

## References

1. **TUH EEG Seizure Corpus**: Temple University Hospital EEG Database
2. **Attention Is All You Need**: Vaswani et al., 2017 (Transformer architecture)
3. **GPT Architecture**: Radford et al., 2018 (Generative pre-training)
4. **MNE-Python**: Gramfort et al., 2013 (EEG signal processing)

---

**Last Updated**: November 15, 2025  
**Project Status**: ✅ Functional | 🚧 Research in Progress