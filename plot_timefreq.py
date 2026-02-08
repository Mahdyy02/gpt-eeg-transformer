from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy import signal

"""
plot_timefreq.py

Plot time-frequency representations (spectrograms) for all channels of patient.
Frequencies more prominent during seizures are colored differently from background frequencies.
"""

def read_csv_annotations(csv_path: Path) -> pd.DataFrame | None:
    """Read a single CSV annotation file."""
    try:
        df = pd.read_csv(csv_path, comment='#', on_bad_lines='skip')
        return df if not df.empty else None
    except Exception as e:
        try:
            df = pd.read_csv(csv_path, comment='#', on_bad_lines='skip', engine='python')
            return df if not df.empty else None
        except Exception as e2:
            print(f"  WARNING: failed to read CSV {csv_path.name}: {e2}")
            return None


def get_bipolar_signal(raw, ch_bipolar: str):
    """
    Create a bipolar signal (e.g., 'FP1-F7') from two monopolar channels in EDF.
    Returns (data, times) or None if electrodes are missing.
    """
    ch_bipolar = ch_bipolar.strip().upper()
    if '-' not in ch_bipolar:
        return None

    ch1, ch2 = ch_bipolar.split('-')
    ch1, ch2 = ch1.strip(), ch2.strip()

    # Helper to match monopolar channel names in EDF
    def match_channel(name):
        for c in raw.ch_names:
            if name in c.upper().replace('EEG', '').replace('_', '').replace(' ', ''):
                return c
        return None

    ch1_full = match_channel(ch1)
    ch2_full = match_channel(ch2)

    if not ch1_full or not ch2_full:
        print(f"⚠️ Missing electrodes for {ch_bipolar}: {ch1_full}, {ch2_full}")
        return None

    data = raw.get_data(picks=[ch1_full])[0] - raw.get_data(picks=[ch2_full])[0]
    return data, raw.times


def compute_spectrogram(data, sfreq, nperseg=256):
    """Compute spectrogram using STFT."""
    f, t, Sxx = signal.spectrogram(data, fs=sfreq, nperseg=nperseg, 
                                     noverlap=nperseg//2, scaling='density')
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


def identify_seizure_frequencies(data, times, annotations, sfreq, nperseg=256):
    """
    Identify which frequencies are more prominent during seizures vs background.
    Returns a frequency mask indicating seizure-dominant frequencies.
    """
    # Compute full spectrogram
    f, t_spec, Sxx_db = compute_spectrogram(data, sfreq, nperseg)
    
    # Create masks for seizure and background periods
    seizure_mask = np.zeros(len(times), dtype=bool)
    background_mask = np.zeros(len(times), dtype=bool)
    
    for _, row in annotations.iterrows():
        start_time = row['start_time']
        stop_time = row['stop_time']
        label = row['label']
        
        time_mask = (times >= start_time) & (times <= stop_time)
        
        if label in ['cpsz', 'spsz', 'gnsz', 'tcsz', 'fnsz', 'absz']:
            seizure_mask |= time_mask
        elif label == 'bckg':
            background_mask |= time_mask
    
    # Map time masks to spectrogram time bins
    seizure_spec_mask = np.zeros(len(t_spec), dtype=bool)
    background_spec_mask = np.zeros(len(t_spec), dtype=bool)
    
    for i, t_val in enumerate(t_spec):
        # Find closest time index
        closest_idx = np.argmin(np.abs(times - t_val))
        seizure_spec_mask[i] = seizure_mask[closest_idx]
        background_spec_mask[i] = background_mask[closest_idx]
    
    # Compute average power for each frequency during seizure and background
    if np.any(seizure_spec_mask):
        seizure_power = np.mean(Sxx_db[:, seizure_spec_mask], axis=1)
    else:
        seizure_power = np.zeros(len(f))
    
    if np.any(background_spec_mask):
        background_power = np.mean(Sxx_db[:, background_spec_mask], axis=1)
    else:
        background_power = np.zeros(len(f))
    
    # Identify seizure-dominant frequencies (higher power during seizure)
    power_diff = seizure_power - background_power
    seizure_freq_mask = power_diff > 0
    
    return f, t_spec, Sxx_db, seizure_freq_mask


def plot_timefreq_channel(patient_name: str, session_name: str, channel_name: str,
                          data: np.ndarray, times: np.ndarray, annotations: pd.DataFrame,
                          sfreq: float, outdir: Path):
    """Plot time-frequency representation with seizure-specific coloring."""
    
    # Identify seizure frequencies
    f, t_spec, Sxx_db, seizure_freq_mask = identify_seizure_frequencies(
        data, times, annotations, sfreq
    )
    
    # Limit frequency range to 0-50 Hz (most relevant for EEG)
    freq_mask = f <= 50
    f = f[freq_mask]
    Sxx_db = Sxx_db[freq_mask, :]
    seizure_freq_mask = seizure_freq_mask[freq_mask]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    
    # Create custom colormap: red for seizure frequencies, blue for background
    # We'll create a modified spectrogram where seizure freqs are highlighted
    Sxx_normalized = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db) + 1e-10)
    
    # Create RGB image
    rgb_image = np.zeros((len(f), len(t_spec), 3))
    
    for i in range(len(f)):
        if seizure_freq_mask[i]:
            # Seizure-dominant frequency: red scale
            rgb_image[i, :, 0] = Sxx_normalized[i, :]  # Red channel
            rgb_image[i, :, 1] = Sxx_normalized[i, :] * 0.3  # Green channel (dim)
            rgb_image[i, :, 2] = Sxx_normalized[i, :] * 0.3  # Blue channel (dim)
        else:
            # Background-dominant frequency: blue scale
            rgb_image[i, :, 0] = Sxx_normalized[i, :] * 0.3  # Red channel (dim)
            rgb_image[i, :, 1] = Sxx_normalized[i, :] * 0.5  # Green channel (medium)
            rgb_image[i, :, 2] = Sxx_normalized[i, :]  # Blue channel
    
    # Plot spectrogram
    extent = [t_spec[0], t_spec[-1], f[0], f[-1]]
    ax1.imshow(rgb_image, aspect='auto', origin='lower', extent=extent, interpolation='bilinear')
    
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title(f"{patient_name} — {session_name} — {channel_name}\n"
                  f"Time-Frequency Analysis (Red=Seizure frequencies, Blue=Background frequencies)")
    ax1.grid(True, alpha=0.2, color='white', linewidth=0.5)
    
    # Add frequency band annotations
    bands = {
        'Delta (0.5-4 Hz)': (0.5, 4),
        'Theta (4-8 Hz)': (4, 8),
        'Alpha (8-13 Hz)': (8, 13),
        'Beta (13-30 Hz)': (13, 30),
        'Gamma (30-50 Hz)': (30, 50)
    }
    
    for band_name, (low, high) in bands.items():
        ax1.axhline(y=low, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.text(t_spec[-1] * 0.98, (low + high) / 2, band_name, 
                color='white', fontsize=8, ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    # Plot phase timeline below
    seizure_occurred = False
    for _, row in annotations.iterrows():
        start_time = row['start_time']
        stop_time = row['stop_time']
        label = row['label']
        
        if label in ['cpsz', 'spsz', 'gnsz', 'tcsz', 'fnsz', 'absz']:
            color = '#DC143C'
            phase_label = 'Seizure'
            seizure_occurred = True
        elif label == 'bckg':
            color = '#1E90FF' if seizure_occurred else '#FF8C00'
            phase_label = 'Post-ictal' if seizure_occurred else 'Pre-ictal'
        else:
            color = '#808080'
            phase_label = label
        
        ax2.axvspan(start_time, stop_time, color=color, alpha=0.6, label=phase_label)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Phase')
    ax2.set_yticks([])
    ax2.set_xlim([t_spec[0], t_spec[-1]])
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # Remove duplicate labels
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    safe_channel_name = channel_name.replace(' ', '_').replace('/', '-')
    outpath = outdir / f"{patient_name}_{session_name}_{safe_channel_name}_timefreq.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {safe_channel_name}_timefreq")


def process_session_timefreq(patient_name: str, csv_path: Path, edf_path: Path, outdir: Path):
    """Process a single session and generate time-frequency plots for all channels."""
    
    # Read CSV annotations
    df_csv = read_csv_annotations(csv_path)
    if df_csv is None or df_csv.empty:
        print(f"❌ No annotations found in {csv_path.name}")
        return
    
    # Load EDF file
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as e:
        print(f"❌ Failed to load EDF {edf_path.name}: {e}")
        return
    
    sfreq = raw.info.get('sfreq', 250)
    times = raw.times
    
    # Get all unique channels from CSV
    channels = df_csv['channel'].unique()
    print(f"\n📊 Processing {csv_path.name}: {len(channels)} channels")
    
    session_name = csv_path.stem  # e.g., aaaaadpj_s003_t000
    
    # Plot each channel
    for channel_name in channels:
        # Get annotations for this specific channel
        channel_annotations = df_csv[df_csv['channel'] == channel_name].copy().sort_values('start_time')
        
        if channel_annotations.empty:
            continue
        
        # Reconstruct bipolar signal from EDF
        result = get_bipolar_signal(raw, channel_name)
        if result is None:
            print(f"  ⚠️ Could not reconstruct channel {channel_name}, skipping")
            continue
        
        data, edf_times = result
        
        try:
            plot_timefreq_channel(patient_name, session_name, channel_name,
                                 data, edf_times, channel_annotations, sfreq, outdir)
        except Exception as e:
            print(f"  ❌ Error plotting {channel_name}: {e}")


def main():
    # Patient folder
    patient_name = "aaaaaalq"
    patient_folder = Path(__file__).parent / patient_name
    
    if not patient_folder.exists():
        print(f"❌ Patient folder not found: {patient_folder}")
        return
    
    # Output directory
    outdir = Path("plots") / f"{patient_name}_timefreq"
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Time-Frequency Analysis for patient: {patient_name}")
    print(f"{'='*60}")
    
    # Find all CSV files in patient folder
    csv_files = sorted(patient_folder.rglob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.endswith("_bi")]
    
    print(f"\nFound {len(csv_files)} CSV files")
    
    for csv_path in csv_files:
        # Find corresponding EDF file
        edf_path = csv_path.with_suffix('.edf')
        
        if not edf_path.exists():
            print(f"⚠️ No EDF file found for {csv_path.name}, skipping")
            continue
        
        session_folder = csv_path.parent
        print(f"\n{'─'*60}")
        print(f"📁 Session: {session_folder.parent.name}/{session_folder.name}")
        
        try:
            process_session_timefreq(patient_name, csv_path, edf_path, outdir)
        except Exception as e:
            print(f"❌ Error processing {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✅ All time-frequency plots saved to: {outdir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
