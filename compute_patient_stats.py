import numpy as np
import pandas as pd
import mne
from pathlib import Path
from collections import defaultdict

# Reuse your existing functions
from train_seizure_prediction import (
    parse_annotations,
    get_seizure_windows,
    load_eeg_session
)

# ---------------- Config ----------------
segment_duration = 30.0
prediction_horizon = 30.0
sampling_rate = 250


# ---------------- Stats Function ----------------
def compute_patient_statistics(patient_dir):
    """
    Compute dataset statistics for one patient (TUH-compatible).
    Includes recursive EDF discovery + debugging messages.
    """
    
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name
    
    print(f"\n{'='*70}")
    print(f"Processing Patient: {patient_id}")
    print(f"{'='*70}")
    
    # --------------------------------------------------
    # Recursive discovery of EDF files (TUH structure)
    # --------------------------------------------------
    
    print("\nSearching for EDF files recursively...")
    
    edf_files = list(patient_dir.rglob("*.edf"))
    
    if len(edf_files) == 0:
        print("No EDF files found in this patient folder.")
        return {
            "Patient": patient_id,
            "Sessions": 0,
            "Recording Hours": 0,
            "Seizures": 0,
            "Segments": 0,
            "Pre-ictal %": 0,
            "Sampling Freq (Hz)": sampling_rate,
            "Channels": 0
        }
    
    print(f"Found {len(edf_files)} EDF files.")
    
    # Pair EDF with CSV
    session_files = []
    
    for edf_path in edf_files:
        
        csv_path = edf_path.with_suffix(".csv")
        
        # Skip if CSV missing
        if not csv_path.exists():
            print(f"   Missing CSV for: {edf_path.name}")
            continue
        
        # Skip Windows metadata artifacts
        if "Zone.Identifier" in csv_path.name:
            continue
        
        session_files.append((edf_path, csv_path))
    
    print(f"\nValid EDF+CSV sessions: {len(session_files)}")
    
    if len(session_files) == 0:
        print("No valid EDF/CSV pairs found.")
    
    # --------------------------------------------------
    # First pass: metadata extraction
    # --------------------------------------------------
    
    max_channels = 0
    total_duration_sec = 0
    total_seizures = 0
    
    print("\nExtracting metadata...")
    
    for i, (edf_path, csv_path) in enumerate(session_files):
        
        print(f"\nSession {i+1}/{len(session_files)}")
        print(f"EDF: {edf_path}")
        print(f"CSV: {csv_path}")
        
        try:
            # Load header only (fast)
            raw = mne.io.read_raw_edf(
                str(edf_path),
                preload=False,
                verbose=False
            )
            
            # Channels
            n_channels = len(raw.ch_names)
            max_channels = max(max_channels, n_channels)
            
            # Duration
            duration_sec = raw.n_times / raw.info['sfreq']
            total_duration_sec += duration_sec
            
            print(f"   Channels: {n_channels}")
            print(f"   Duration: {duration_sec/60:.2f} min")
            
            # Seizures
            annotations = parse_annotations(csv_path)
            seizure_windows = get_seizure_windows(annotations)
            
            n_seizures = len(seizure_windows)
            total_seizures += n_seizures
            
            print(f"   Seizures: {n_seizures}")
        
        except Exception as e:
            print(f"Error reading session: {e}")
    
    print(f"\nTotal recording hours: {total_duration_sec/3600:.2f}")
    print(f"Total seizures: {total_seizures}")
    print(f"Max channels: {max_channels}")
    
    # --------------------------------------------------
    # Second pass: segment extraction
    # --------------------------------------------------
    
    print("\nExtracting segments...")
    
    all_segments = []
    
    for i, (edf_path, csv_path) in enumerate(session_files):
        
        print(f"\nSegmenting session {i+1}/{len(session_files)}...")
        
        try:
            segments = load_eeg_session(
                edf_path,
                csv_path,
                segment_duration,
                prediction_horizon,
                sampling_rate,
                target_channels=max_channels
            )
            
            print(f"   Segments extracted: {len(segments)}")
            
            all_segments.extend(segments)
        
        except Exception as e:
            print(f"Segmentation failed: {e}")
    
    # --------------------------------------------------
    # Segment statistics
    # --------------------------------------------------
    
    total_segments = len(all_segments)
    
    pos_segments = sum(1 for _, y in all_segments if y == 1)
    neg_segments = total_segments - pos_segments
    
    preictal_ratio = (
        pos_segments / total_segments * 100
        if total_segments > 0 else 0
    )
    
    # Convert duration
    total_hours = total_duration_sec / 3600
    
    print(f"\nSegment Summary")
    print(f"Total segments: {total_segments}")
    print(f"Pre-ictal: {pos_segments}")
    print(f"Non-pre-ictal: {neg_segments}")
    print(f"Pre-ictal ratio: {preictal_ratio:.2f}%")
    
    stats = {
        "Patient": patient_id,
        "Sessions": len(session_files),
        "Recording Hours": round(total_hours, 2),
        "Seizures": total_seizures,
        "Segments": total_segments,
        "Pre-ictal %": round(preictal_ratio, 2),
        "Sampling Freq (Hz)": sampling_rate,
        "Channels": max_channels
    }
    
    return stats


# ---------------- Multi-Patient Runner ----------------
def compute_all_patients_stats(root_dir, output_csv="patient_statistics.csv"):
    """
    Compute stats for all patients and save CSV.
    """
    
    root_dir = Path(root_dir)
    
    print(f"\nScanning root directory: {root_dir}")
    
    # Only keep real patient folders
    patient_dirs = [
        d for d in root_dir.iterdir()
        if d.is_dir()
        and not any(x in d.name.lower() for x in ["plot", "timefreq", "result"])
    ]
    
    print(f"Patients detected: {len(patient_dirs)}")
    
    all_stats = []
    
    for patient_dir in patient_dirs:
        stats = compute_patient_statistics(patient_dir)
        all_stats.append(stats)
    
    df = pd.DataFrame(all_stats)
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*70}")
    print("FINAL DATASET TABLE")
    print(f"{'='*70}")
    print(df)
    
    print(f"\nSaved to: {output_csv}")
    
    return df


# ---------------- Main ----------------
if __name__ == "__main__":
    
    # Root directory containing patient folders
    root_data_dir = "aaaaaaac"
    
    compute_all_patients_stats(root_data_dir)
