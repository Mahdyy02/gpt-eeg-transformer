from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne

"""
plot_aaaaadpj_all_channels.py

Plot all channels from all CSV annotation files for patient aaaaadpj.
Each session (CSV file) will be plotted with all its channels.
Uses the same color coding as plot.py: pre-ictal, ictal, post-ictal phases.
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


def plot_session_all_channels(patient_name: str, session_folder: Path, csv_path: Path, edf_path: Path, outdir: Path):
    """Plot all channels from a single session CSV file."""
    
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
    
    # Color scheme matching plot.py
    colors = {
        'bckg': {'pre': '#FF8C00', 'post': '#1E90FF'},
        'cpsz': '#DC143C', 'fnsz': '#DC143C', 'gnsz': '#DC143C', 
        'tcsz': '#DC143C', 'spsz': '#DC143C', 'absz': '#DC143C',
    }
    
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
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        
        seizure_occurred = False
        legend_labels = set()
        
        for idx, row in channel_annotations.iterrows():
            start_time = row['start_time']
            stop_time = row['stop_time']
            label = row['label']
            
            mask = (edf_times >= start_time) & (edf_times <= stop_time)
            segment_times = edf_times[mask]
            segment_data = data[mask]
            
            if len(segment_times) == 0:
                continue
            
            # Determine color and phase label
            if label in ['cpsz', 'spsz', 'gnsz', 'tcsz', 'fnsz', 'absz']:
                color = colors.get(label, '#DC143C')
                phase_label = 'Seizure (ictal)'
                seizure_occurred = True
            elif label == 'bckg':
                color = colors['bckg']['post'] if seizure_occurred else colors['bckg']['pre']
                phase_label = 'Post-ictal' if seizure_occurred else 'Pre-ictal'
            else:
                color = '#808080'
                phase_label = label
            
            # Plot segment
            if phase_label not in legend_labels:
                ax.plot(segment_times, segment_data, color=color, linewidth=0.8, label=phase_label)
                legend_labels.add(phase_label)
            else:
                ax.plot(segment_times, segment_data, color=color, linewidth=0.8)
        
        # Configure plot
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{channel_name} (μV)')
        ax.set_title(f"{patient_name} — {session_name} — {channel_name}")
        ax.grid(True, alpha=0.3)
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        safe_channel_name = channel_name.replace(' ', '_').replace('/', '-')
        outpath = outdir / f"{patient_name}_{session_name}_{safe_channel_name}.png"
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
        print(f"  ✅ Saved: {safe_channel_name}")


def main():
    # Patient folder
    patient_folder = Path(__file__).parent / "aaaaadpj"
    patient_name = "aaaaadpj"
    
    if not patient_folder.exists():
        print(f"❌ Patient folder not found: {patient_folder}")
        return
    
    # Output directory
    outdir = Path("plots") / patient_name
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Plotting all channels for patient: {patient_name}")
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
            plot_session_all_channels(patient_name, session_folder, csv_path, edf_path, outdir)
        except Exception as e:
            print(f"❌ Error processing {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✅ All plots saved to: {outdir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
