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
- Compute FAR (False Alarm Rate) and Pdelay (Prediction Delay)
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
        # Use segment end time to align with training labels:
        # label is based on seizure within [segment_end, segment_end + prediction_horizon]
        time = (start_sample + segment_samples) / sampling_rate
        
        prediction, confidence = predict_segment(model, segment, device)
        
        predictions.append(prediction)
        confidences.append(confidence)
        times.append(time)
    
    return times, predictions, confidences, seizure_windows, duration


def identify_alarm_periods(times, predictions):
    """
    Identify continuous alarm periods from predictions.
    
    An alarm period is a continuous segment where majority of predictions are positive (seizure=1).
    Uses a sliding window to determine if a period should continue.
    
    Args:
        times: list of prediction timestamps
        predictions: list of binary predictions (0 or 1)
    
    Returns:
        alarm_periods: list of dicts with 'start_time', 'end_time', 'start_idx', 'end_idx'
    """
    alarm_periods = []
    
    if len(times) == 0:
        return alarm_periods
    
    # Define window size for grouping predictions
    window_size = 5
    
    i = 0
    while i < len(predictions):
        # Check if current position starts an alarm period
        if predictions[i] == 1:
            # Start of potential alarm period
            period_start_idx = i
            period_start_time = times[i]
            
            # Extend the period while there are more 1s than 0s in sliding window
            j = i
            while j < len(predictions):
                # Look ahead at next window_size predictions
                window_end = min(j + window_size, len(predictions))
                window_preds = predictions[j:window_end]
                
                # Count seizure vs non-seizure predictions
                seizure_count = sum(window_preds)
                non_seizure_count = len(window_preds) - seizure_count
                
                # Continue period if more seizure predictions
                if seizure_count > non_seizure_count:
                    j += 1
                else:
                    break
            
            # Period ends at j-1
            if j > i:  # Valid alarm period found
                period_end_idx = j - 1
                period_end_time = times[period_end_idx]
                
                alarm_periods.append({
                    'start_time': period_start_time,
                    'end_time': period_end_time,
                    'start_idx': period_start_idx,
                    'end_idx': period_end_idx,
                    'alarm_count': sum(predictions[period_start_idx:period_end_idx + 1]),
                    'total_count': period_end_idx - period_start_idx + 1
                })
                
                i = j  # Move to end of this period
            else:
                i += 1
        else:
            i += 1
    
    return alarm_periods


def _period_overlaps_preictal_window(period, sz_start, prediction_horizon):
    """
    Check if an alarm period overlaps the pre-ictal window
    [sz_start - prediction_horizon, sz_start].
    """
    window_start = sz_start - prediction_horizon
    window_end = sz_start
    return period['end_time'] >= window_start and period['start_time'] <= window_end


def compute_false_alarm_rate(times, predictions, seizure_windows, prediction_horizon, total_duration):
    """
    Compute False Alarm Rate (FAR) based on alarm periods, not individual predictions.
    
    FAR = Number of false alarm periods / Total recording duration (hours)
    
    An alarm period is a continuous segment where the model predicts seizures.
    A false alarm period is one that does NOT correctly predict any seizure.
    """
    
    # Step 1: Identify all alarm periods
    alarm_periods = identify_alarm_periods(times, predictions)
    
    # Step 2: Classify each alarm period as true alarm or false alarm
    false_alarm_periods = 0
    true_alarm_periods = 0
    
    for period in alarm_periods:
        # Check if this alarm period correctly predicts any seizure
        is_true_alarm = False
        
        for sz_start, sz_stop in seizure_windows:
            # Alarm is valid if it overlaps the pre-ictal window before seizure onset
            if _period_overlaps_preictal_window(period, sz_start, prediction_horizon):
                is_true_alarm = True
                break
        
        if is_true_alarm:
            true_alarm_periods += 1
        else:
            false_alarm_periods += 1
    
    # Step 3: Calculate FAR
    total_hours = total_duration / 3600.0
    far = false_alarm_periods / total_hours if total_hours > 0 else 0
    
    print(f"\n  Alarm Period Analysis for FAR:")
    print(f"    Total alarm periods detected: {len(alarm_periods)}")
    print(f"    True alarm periods (correctly predict seizures): {true_alarm_periods}")
    print(f"    False alarm periods (do not predict seizures): {false_alarm_periods}")
    print(f"    Recording duration: {total_hours:.4f} hours ({total_duration:.1f} seconds)")
    
    return far, false_alarm_periods, total_hours


def compute_prediction_delay(times, predictions, seizure_windows, prediction_horizon):
    """
    Compute maximum prediction delay (Pdelay) for correctly predicted seizures.
    
    Pdelay is calculated based on continuous alarm periods:
    - Group consecutive predictions into periods
    - A period is classified as "alarm period" if it contains more seizure predictions than non-seizure
    - For each seizure, find the earliest alarm period that precedes it
    - Delay = seizure_onset - start_of_alarm_period
    - Return the maximum delay across all detected seizures
    
    Args:
        times: list of prediction timestamps
        predictions: list of binary predictions (0 or 1)
        seizure_windows: list of (start, stop) tuples for actual seizures
        prediction_horizon: maximum time before seizure to consider valid prediction
    
    Returns:
        max_delay: maximum prediction delay in seconds
        avg_delay: average delay across detected seizures in seconds
        detected_seizures: number of seizures with at least one preceding alarm period
        individual_delays: list of delays for each detected seizure
        seizure_details: detailed information for each seizure
    """
    
    # Step 1: Identify alarm periods
    alarm_periods = identify_alarm_periods(times, predictions)
    
    if len(alarm_periods) == 0:
        print("\n  No alarm periods found!")
        return 0, 0, 0, [], []
    
    print(f"\n  Found {len(alarm_periods)} alarm periods for Pdelay calculation:")
    for idx, period in enumerate(alarm_periods):
        duration = period['end_time'] - period['start_time']
        alarm_ratio = period['alarm_count'] / period['total_count'] * 100
        print(f"    Period {idx+1}: {period['start_time']:.1f}s - {period['end_time']:.1f}s "
              f"(duration: {duration:.1f}s, alarm ratio: {alarm_ratio:.1f}%)")
    
    # Step 2: Match alarm periods to seizures
    delays = []
    detected_seizures = 0
    seizure_details = []
    
    for sz_idx, (sz_start, sz_stop) in enumerate(seizure_windows):
        # Find all alarm periods that could predict this seizure
        valid_periods = []
        
        for period in alarm_periods:
            # Alarm period must overlap the pre-ictal window
            if _period_overlaps_preictal_window(period, sz_start, prediction_horizon):
                valid_periods.append(period)
        
        if valid_periods:
            # Found at least one alarm period for this seizure
            detected_seizures += 1
            
            # Find earliest alarm time within the pre-ictal window
            window_start = sz_start - prediction_horizon
            earliest_period = min(valid_periods, key=lambda p: max(p['start_time'], window_start))
            alarm_time = max(earliest_period['start_time'], window_start)
            delay = sz_start - alarm_time
            delays.append(delay)
            
            seizure_details.append({
                'seizure_idx': sz_idx + 1,
                'seizure_start': sz_start,
                'alarm_period_start': earliest_period['start_time'],
                'alarm_period_end': earliest_period['end_time'],
                'alarm_time_within_window': alarm_time,
                'delay': delay,
                'num_matching_periods': len(valid_periods)
            })
            
            print(f"\n  Seizure {sz_idx+1} at {sz_start:.1f}s:")
            print(f"    Earliest alarm period: {earliest_period['start_time']:.1f}s - {earliest_period['end_time']:.1f}s")
            print(f"    Alarm time used: {alarm_time:.1f}s (within pre-ictal window)")
            print(f"    Prediction delay: {delay:.2f}s ({delay/3600:.4f} hours)")
            print(f"    Total matching periods: {len(valid_periods)}")
        else:
            print(f"\n  Seizure {sz_idx+1} at {sz_start:.1f}s: NOT DETECTED (no alarm periods found)")
    
    # Step 3: Calculate summary statistics
    if delays:
        max_delay = max(delays)
        avg_delay = np.mean(delays)
    else:
        max_delay = 0
        avg_delay = 0
    
    return max_delay, avg_delay, detected_seizures, delays, seizure_details


def compute_metrics(times, predictions, confidences, seizure_windows, prediction_horizon, total_duration):
    """Compute comprehensive metrics including precision, recall, F1, FAR, and Pdelay."""
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
    
    # Compute basic metrics
    tp = ((predictions == 1) & (true_labels == 1)).sum()
    fp = ((predictions == 1) & (true_labels == 0)).sum()
    fn = ((predictions == 0) & (true_labels == 1)).sum()
    tn = ((predictions == 0) & (true_labels == 0)).sum()
    
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Compute False Alarm Rate (using alarm periods)
    far, num_false_alarms, total_hours = compute_false_alarm_rate(
        times, predictions, seizure_windows, prediction_horizon, total_duration
    )
    
    # Compute Prediction Delay (using alarm periods)
    max_delay, avg_delay, detected_seizures, individual_delays, seizure_details = compute_prediction_delay(
        times, predictions, seizure_windows, prediction_horizon
    )
    
    # Calculate sensitivity (percentage of seizures detected)
    total_seizures = len(seizure_windows)
    sensitivity = (detected_seizures / total_seizures * 100) if total_seizures > 0 else 0
    
    print(f"\n{'='*70}")
    print("PERFORMANCE METRICS")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy:.4f} ({tp + tn}/{len(predictions)})")
    print(f"Precision: {precision:.4f} (TP={tp}, FP={fp})")
    print(f"Recall:    {recall:.4f} (TP={tp}, FN={fn})")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'-'*70}")
    print(f"FALSE ALARM RATE (FAR) - Based on Alarm Periods:")
    print(f"  Total false alarm periods: {num_false_alarms}")
    print(f"  Recording duration: {total_hours:.4f} hours ({total_duration:.1f} seconds)")
    print(f"  FAR: {far:.2f} false alarm periods per hour")
    print(f"{'-'*70}")
    print(f"PREDICTION DELAY (Pdelay) - Based on Alarm Periods:")
    print(f"  Total seizures: {total_seizures}")
    print(f"  Seizures detected: {detected_seizures} ({sensitivity:.1f}%)")
    print(f"  Maximum delay (Pdelay): {max_delay:.2f} seconds ({max_delay/3600:.4f} hours)")
    print(f"  Average delay: {avg_delay:.2f} seconds ({avg_delay/3600:.4f} hours)")
    
    if individual_delays:
        print(f"  Individual delays: {[f'{d:.2f}s' for d in individual_delays]}")
    
    print(f"{'='*70}\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'far': far,
        'num_false_alarms': num_false_alarms,
        'total_hours': total_hours,
        'max_delay_seconds': max_delay,
        'max_delay_hours': max_delay / 3600,
        'avg_delay_seconds': avg_delay,
        'avg_delay_hours': avg_delay / 3600,
        'total_seizures': total_seizures,
        'detected_seizures': detected_seizures,
        'sensitivity': sensitivity,
        'individual_delays': individual_delays,
        'seizure_details': seizure_details
    }


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
    
    # Compute comprehensive metrics
    metrics = compute_metrics(
        times, predictions, confidences, seizure_windows, 
        hyperparams['prediction_horizon'], duration
    )
    
    # Plot
    plot_predictions(times, predictions, confidences, seizure_windows, duration, args.output)
    
    print(f"\n✅ Inference complete!")
    print(f"\nKey Results Summary:")
    print(f"  - F1 Score: {metrics['f1']:.4f}")
    print(f"  - FAR: {metrics['far']:.2f} alarm periods/hour")
    print(f"  - Max Delay (Pdelay): {metrics['max_delay_hours']:.4f} hours ({metrics['max_delay_seconds']:.2f} seconds)")
    print(f"  - Sensitivity: {metrics['sensitivity']:.1f}% ({metrics['detected_seizures']}/{metrics['total_seizures']} seizures)")


if __name__ == "__main__":
    main()
