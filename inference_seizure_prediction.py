# inference_seizure_prediction.py
import torch
import torch.nn as nn
import numpy as np
import mne
from pathlib import Path
import matplotlib.pyplot as plt
from train_seizure_prediction import TransformerEncoderFineTune, normalize_segment, parse_annotations, get_seizure_windows

"""
Inference script for seizure prediction model.
- Load trained model
- Test on new 30-second EEG segments
- Visualize predictions with confidence scores
"""

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    hyperparams = checkpoint['hyperparameters']
    
    # Get vocab_size from checkpoint (default 512 if not present)
    vocab_size = checkpoint.get('vocab_size', 512)
    
    model = TransformerEncoderFineTune(
        n_channels=hyperparams['n_channels'],
        n_embd=hyperparams['n_embd'],
        n_head=hyperparams['n_head'],
        n_layer=hyperparams['n_layer'],
        max_seq_len=hyperparams['max_seq_len'],
        dropout=hyperparams['dropout'],
        vocab_size=vocab_size,
        pretrained_path=None  # Don't reload pretrained during inference
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Segment duration: {hyperparams['segment_duration']}s")
    print(f"Prediction horizon: {hyperparams['prediction_horizon']}s")
    print(f"Sampling rate: {hyperparams['sampling_rate']}Hz\n")
    
    return model, hyperparams


def predict_segment(model, segment, device='cuda'):
    """
    Predict seizure risk for a single segment.
    
    Args:
        model: trained model
        segment: (n_channels, seq_len) numpy array
        
    Returns:
        prediction: 0 (no seizure) or 1 (seizure)
        confidence: probability of seizure
    """
    # Normalize
    segment = normalize_segment(segment)
    
    # Convert to torch and add batch dimension
    segment_tensor = torch.from_numpy(segment).float().unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(segment_tensor)
        probs = torch.softmax(logits, dim=-1)
        prediction = logits.argmax(dim=-1).item()
        confidence = probs[0, 1].item()  # Probability of seizure
    
    return prediction, confidence


def test_on_file(model, edf_path, csv_path, hyperparams, device='cuda'):
    """Test model on entire EEG file and visualize predictions."""
    print(f"\nTesting on {edf_path.name}...")
    
    # Load EEG
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    
    # Resample if needed
    sampling_rate = hyperparams['sampling_rate']
    if raw.info['sfreq'] != sampling_rate:
        raw.resample(sampling_rate, verbose=False)
    
    data = raw.get_data()
    
    # Pad/truncate to match model's expected channel count
    target_channels = hyperparams['n_channels']
    if data.shape[0] < target_channels:
        # Pad with zeros
        padding = np.zeros((target_channels - data.shape[0], data.shape[1]))
        data = np.vstack([data, padding])
        print(f"  Padded from {raw.get_data().shape[0]} to {target_channels} channels")
    elif data.shape[0] > target_channels:
        # Truncate
        data = data[:target_channels, :]
        print(f"  Truncated from {raw.get_data().shape[0]} to {target_channels} channels")
    
    duration = data.shape[1] / sampling_rate
    
    # Parse annotations
    annotations = parse_annotations(csv_path)
    seizure_windows = get_seizure_windows(annotations)
    
    print(f"Duration: {duration:.1f}s")
    print(f"Found {len(seizure_windows)} seizure windows: {seizure_windows}")
    
    # Create segments
    segment_duration = hyperparams['segment_duration']
    segment_samples = int(segment_duration * sampling_rate)
    step_size = segment_samples // 4  # 75% overlap for smooth prediction
    
    predictions = []
    confidences = []
    times = []
    
    for start_sample in range(0, data.shape[1] - segment_samples, step_size):
        segment = data[:, start_sample:start_sample + segment_samples]
        time = start_sample / sampling_rate
        
        prediction, confidence = predict_segment(model, segment, device)
        
        predictions.append(prediction)
        confidences.append(confidence)
        times.append(time)
    
    return times, predictions, confidences, seizure_windows, duration


def plot_predictions(times, predictions, confidences, seizure_windows, duration, save_path='predictions.png'):
    """Plot prediction timeline with ground truth seizure windows."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    # Plot 1: Binary predictions
    colors = ['green' if p == 0 else 'red' for p in predictions]
    axes[0].scatter(times, predictions, c=colors, alpha=0.6, s=20)
    axes[0].set_ylabel('Prediction\n(0=No Seizure, 1=Seizure)', fontsize=11)
    axes[0].set_ylim(-0.2, 1.2)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['No Seizure', 'Seizure'])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Seizure Predictions Over Time', fontsize=14, fontweight='bold')
    
    # Add ground truth seizure windows
    for sz_start, sz_stop in seizure_windows:
        axes[0].axvspan(sz_start, sz_stop, alpha=0.3, color='orange', label='True Seizure')
    
    # Remove duplicate labels
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Plot 2: Confidence scores
    axes[1].plot(times, confidences, linewidth=1.5, color='#2E86AB')
    axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Decision Threshold')
    axes[1].fill_between(times, 0, confidences, alpha=0.3, color='#2E86AB')
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_ylabel('Seizure Probability', fontsize=11)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Seizure Prediction Confidence', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right')
    
    # Add ground truth seizure windows
    for sz_start, sz_stop in seizure_windows:
        axes[1].axvspan(sz_start, sz_stop, alpha=0.3, color='orange')
    
    plt.xlim(0, duration)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to {save_path}")
    plt.show()


def compute_metrics(times, predictions, confidences, seizure_windows, prediction_horizon):
    """Compute precision, recall, F1 score with temporal tolerance."""
    # For each prediction, check if there's a seizure within prediction_horizon
    true_labels = []
    for time in times:
        has_seizure = False
        for sz_start, sz_stop in seizure_windows:
            if sz_start <= time + prediction_horizon and sz_start >= time:
                has_seizure = True
                break
        true_labels.append(int(has_seizure))
    
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    
    # Compute metrics
    tp = ((predictions == 1) & (true_labels == 1)).sum()
    fp = ((predictions == 1) & (true_labels == 0)).sum()
    fn = ((predictions == 0) & (true_labels == 1)).sum()
    tn = ((predictions == 0) & (true_labels == 0)).sum()
    
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*60}")
    print("Performance Metrics:")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f} ({tp + tn}/{len(predictions)})")
    print(f"Precision: {precision:.4f} (TP={tp}, FP={fp})")
    print(f"Recall:    {recall:.4f} (TP={tp}, FN={fn})")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'='*60}\n")
    
    return accuracy, precision, recall, f1


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Seizure prediction inference')
    parser.add_argument('--checkpoint', type=str, default='seizure_prediction_transformer.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--edf', type=str, required=True,
                        help='Path to EDF file')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to CSV annotation file')
    parser.add_argument('--output', type=str, default='predictions.png',
                        help='Output plot path')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load model
    model, hyperparams = load_model(args.checkpoint, device)
    
    # Test on file
    times, predictions, confidences, seizure_windows, duration = test_on_file(
        model, Path(args.edf), Path(args.csv), hyperparams, device
    )
    
    # Compute metrics
    compute_metrics(times, predictions, confidences, seizure_windows, hyperparams['prediction_horizon'])
    
    # Plot
    plot_predictions(times, predictions, confidences, seizure_windows, duration, args.output)
    
    print(f"\n✅ Inference complete!")


if __name__ == "__main__":
    main()
