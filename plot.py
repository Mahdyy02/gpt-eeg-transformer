from pathlib import Path
import pandas as pd
import numpy as np
import mne

"""
tuh_eeg.py

Load patient folders under a "tuh_eeg" directory,
read EDF files for each patient, and print the EEG signals per channel.

Usage:
    python tuh_eeg.py [path/to/tuh_eeg_folder]
    python tuh_eeg.py [path/to/tuh_eeg_folder] --plot
    python tuh_eeg.py [path/to/tuh_eeg_folder] --plot --channel EEG_FP1

Creates a dict `patients` mapping patient folder name -> dict with keys:
  - 'edf': EDF DataFrame (channels x samples as columns, indexed by time)
  - 'raw': MNE Raw object for advanced processing
  - 'edf_path': Path to the EDF file
"""

def read_edf_as_dataframe(edf_path: Path) -> pd.DataFrame | None:
    """Read an EDF file and return it as a DataFrame with time index."""
    if not edf_path or not edf_path.exists():
        return None
    if mne is None:
        print("  NOTE: 'mne' not installed; skipping EDF reading. Install with `pip install mne` to enable.")
        return None
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        data = raw.get_data()  # shape (n_channels, n_times)
        times = np.arange(data.shape[1]) / raw.info["sfreq"]
        df = pd.DataFrame(data.T, columns=raw.ch_names)
        df.index = times
        df.index.name = "time_s"
        return df
    except Exception as e:
        print(f"  WARNING: failed to read EDF {edf_path.name}: {e}")
        return None


def read_all_csvs_in_folder(folder: Path) -> pd.DataFrame | None:
    """Read all CSV annotation files in a folder and combine them into a single DataFrame."""
    csv_files = sorted(folder.rglob("*.csv"))
    if not csv_files:
        return None
    
    all_dfs = []
    for csv_path in csv_files:
        try:
            # CSV files have comment lines starting with '#'
            df = pd.read_csv(csv_path, comment='#', on_bad_lines='skip')
            if not df.empty:
                all_dfs.append(df)
        except Exception as e:
            try:
                # Fallback to python engine
                df = pd.read_csv(csv_path, comment='#', on_bad_lines='skip', engine='python')
                if not df.empty:
                    all_dfs.append(df)
            except Exception as e2:
                print(f"  WARNING: failed to read CSV {csv_path.name}: {e2}")
                continue
    
    if not all_dfs:
        return None
    
    # Combine all CSV files
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def load_patients(tuh_dir: Path) -> dict:
    """Load all patient folders and their EDF files and CSV annotations."""
    patients = {}
    if not tuh_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {tuh_dir}")
    
    for p in sorted(tuh_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        print(f"Loading patient folder: {name}")
        
        # Find first .edf file in folder or subfolders (recursive)
        edf_files = sorted(p.rglob("*.edf"))
        # Prefer .edf files that are not tiny companion files
        edf_files = [f for f in edf_files if not f.name.endswith("_bi.edf")]
        edf_path = edf_files[0] if edf_files else None
        
        df_edf = read_edf_as_dataframe(edf_path) if edf_path is not None else None
        
        # Also load the raw MNE object for plotting and exact sfreq
        raw = None
        if edf_path is not None and mne is not None:
            try:
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            except Exception as e:
                print(f"  WARNING: failed to load raw MNE object: {e}")
                raw = None
        
        # Load CSV annotations
        df_csv = read_all_csvs_in_folder(p)
        
        # Create combined dataframe with sample indices
        combined_df = None
        if df_csv is not None and raw is not None:
            sfreq = raw.info.get('sfreq', 250)  # Default to 250 Hz if not available
            combined_df = df_csv.copy()
            combined_df['start_sample'] = (combined_df['start_time'] * sfreq).astype(int)
            combined_df['stop_sample'] = (combined_df['stop_time'] * sfreq).astype(int)

        patients[name] = {
            "edf": df_edf,
            "raw": raw,
            "edf_path": edf_path,
            "csv": df_csv,
            "combined": combined_df
        }
    
    return patients


def print_patient_info(patients: dict):
    """Print information about each patient's EEG signals."""
    for name, d in patients.items():
        print(f"\n{'='*60}")
        print(f"PATIENT: {name}")
        print(f"{'='*60}")
        
        df_edf = d.get("edf")
        edf_path = d.get("edf_path")
        raw = d.get("raw")

        if edf_path is None:
            print(" ❌ No EDF file found")
            continue
        
        print(f" 📁 EDF File: {edf_path.name}")
        
        if df_edf is None:
            print(" ❌ Failed to load EDF data")
            continue
        
        # Print overall info
        print(f" 📊 Shape: {df_edf.shape} (time_points × channels)")
        print(f" ⏱️  Duration: {df_edf.index[-1]:.2f} seconds")
        if raw is not None:
            print(f" 🔊 Sampling Frequency: {raw.info['sfreq']:.2f} Hz")
        
        # Print channel-by-channel information
        print(f"\n 📡 CHANNELS ({len(df_edf.columns)}):")
        print(f" {'-'*58}")
        
        for i, channel in enumerate(df_edf.columns, 1):
            channel_data = df_edf[channel].values
            print(f"\n  [{i}] {channel}")
            print(f"      • Samples: {len(channel_data)}")
            print(f"      • Mean: {np.mean(channel_data):.6f}")
            print(f"      • Std: {np.std(channel_data):.6f}")
            print(f"      • Min: {np.min(channel_data):.6f}")
            print(f"      • Max: {np.max(channel_data):.6f}")
            print(f"      • First 5 values: {channel_data[:5]}")
        
        print(f"\n {'-'*58}")


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


def plot_patient_channels(patient_name: str, patient: dict, outdir: Path | str = "plots", channel_name: str | None = None):
    """Plot EEG channels for the patient using bipolar reconstruction from EDF to match CSV."""
    import matplotlib.pyplot as plt
    import mne

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = patient.get('raw')
    edf_path = patient.get('edf_path')
    combined_df = patient.get('combined')

    if raw is None and edf_path is None:
        print(f"❌ No EDF available for {patient_name}, skipping plot.")
        return None

    # Load raw if not present
    if raw is None and edf_path is not None:
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        except Exception as e:
            print(f"❌ Failed to load EDF for {patient_name}: {e}")
            return None

    if channel_name is None:
        # Default: pick first CSV channel or first EEG channel in EDF
        if combined_df is not None and not combined_df.empty:
            channel_name = combined_df['channel'].iloc[0]
        else:
            ch_candidates = [c for c in raw.ch_names if 'EEG' in c.upper()]
            channel_name = ch_candidates[0] if ch_candidates else raw.ch_names[0]

    # Reconstruct bipolar signal if channel is not in EDF directly
    if channel_name not in raw.ch_names:
        result = get_bipolar_signal(raw, channel_name)
        if result is None:
            print(f"❌ Could not reconstruct channel {channel_name} from EDF electrodes")
            return None
        data, times = result
    else:
        data = raw.get_data(picks=[channel_name])[0]
        times = raw.times

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    if combined_df is not None and not combined_df.empty:
        # Exact match on CSV channel
        norm_csv_channels = combined_df['channel'].str.upper().str.replace(' ', '').str.replace('_', '')
        norm_target = channel_name.upper().replace(' ', '').replace('_', '')
        channel_annotations = combined_df[norm_csv_channels == norm_target].copy().sort_values('start_time')

        print(f"  📍 Channel: {channel_name}, found {len(channel_annotations)} annotation segments")

        if not channel_annotations.empty:
            colors = {
                'bckg': {'pre': '#FF8C00', 'post': '#1E90FF'},
                'cpsz': '#DC143C', 'fnsz': '#DC143C', 'gnsz': '#DC143C', 'tcsz': '#DC143C', 'spsz': '#DC143C', 'absz' : '#DC143C',
            }
            seizure_occurred = False
            legend_labels = set()

            for idx, row in channel_annotations.iterrows():
                start_time = row['start_time']
                stop_time = row['stop_time']
                label = row['label']

                mask = (times >= start_time) & (times <= stop_time)
                segment_times = times[mask]
                segment_data = data[mask]

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

                if phase_label not in legend_labels:
                    ax.plot(segment_times, segment_data, color=color, linewidth=0.8, label=phase_label)
                    legend_labels.add(phase_label)
                else:
                    ax.plot(segment_times, segment_data, color=color, linewidth=0.8)
        else:
            print(f"  ⚠️ No matching annotations found, plotting in black")
            ax.plot(times, data, color='black', linewidth=0.6)
    else:
        print(f"  ⚠️ No combined annotations available")
        ax.plot(times, data, color='black', linewidth=0.6)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'{channel_name} (μV)')
    ax.set_title(f"{patient_name} — {Path(raw.filenames[0]).name if raw.filenames else ''} — {channel_name}")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    outpath = outdir / f"{patient_name}_{channel_name.replace(' ','_').replace('/','-')}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"✅ Saved plot for {patient_name} -> {outpath}")
    return outpath



def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog='tuh_eeg.py',
        description='Load and display EEG signals from TUH EEG dataset'
    )
    parser.add_argument(
        'tuh_dir',
        nargs='?',
        default=str(Path(__file__).parent / "tuh_eeg/train"),
        help='Path to TUH EEG folder containing patient directories'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots for each patient and save to plots/'
    )
    parser.add_argument(
        '--channel',
        default=None,
        help='Specific channel name to plot (default: first EEG channel)'
    )
    args = parser.parse_args()

    base = Path(args.tuh_dir)
    print(f"Loading patients from: {base.absolute()}\n")
    
    patients = load_patients(base)
    print_patient_info(patients)

    if args.plot:
        print(f"\n{'='*60}")
        print("GENERATING PLOTS")
        print(f"{'='*60}\n")
        outdir = Path('plots')
        for name, pdata in patients.items():
            try:
                plot_patient_channels(name, pdata, outdir=outdir, channel_name=args.channel)
            except Exception as e:
                print(f"❌ Failed to plot {name}: {e}")


if __name__ == "__main__":
    main()