from pathlib import Path
import pandas as pd
import numpy as np
import mne
from scipy import signal
from scipy.stats import variation

"""
filter_and_merge.py

Pour chaque session du patient :
1. Détecte les fréquences de bruit (uniformes sur toutes les bandes de fréquence)
2. Applique un filtre notch pour enlever ces bruits
3. Sauvegarde toutes les mesures dans un seul fichier EDF avec annotations par session
"""

def read_csv_annotations(csv_path: Path) -> pd.DataFrame | None:
    """Lit un fichier CSV d'annotations."""
    try:
        df = pd.read_csv(csv_path, comment='#', on_bad_lines='skip')
        return df if not df.empty else None
    except Exception as e:
        try:
            df = pd.read_csv(csv_path, comment='#', on_bad_lines='skip', engine='python')
            return df if not df.empty else None
        except Exception as e2:
            print(f"  WARNING: échec de lecture CSV {csv_path.name}: {e2}")
            return None


def detect_noise_frequencies(raw, max_freq=60, variance_threshold=0.20, power_percentile=80):
    """
    Détecte les fréquences de bruit uniforme sur tous les canaux.
    
    Paramètres:
        raw: objet MNE Raw
        max_freq: fréquence maximale à analyser (Hz)
        variance_threshold: seuil du coefficient de variation pour détecter l'uniformité
        power_percentile: percentile de puissance pour considérer un pic
    
    Retourne:
        Liste des fréquences de bruit détectées
    """
    print(f"  🔍 Détection des fréquences de bruit...")
    
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    
    # Sélectionner uniquement les canaux EEG
    eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    if len(eeg_picks) == 0:
        eeg_picks = range(min(20, data.shape[0]))  # Fallback: premiers 20 canaux
    
    # Calculer le spectre de puissance pour chaque canal EEG
    nperseg = min(int(sfreq * 4), data.shape[1])  # Fenêtre de 4 secondes
    
    all_psds = []
    for i in eeg_picks:
        freqs, psd = signal.welch(data[i], fs=sfreq, nperseg=nperseg)
        all_psds.append(psd)
    
    all_psds = np.array(all_psds)  # shape: (n_channels, n_freqs)
    
    # Ne garder que les fréquences jusqu'à max_freq
    freq_mask = freqs <= max_freq
    freqs = freqs[freq_mask]
    all_psds = all_psds[:, freq_mask]
    
    # Normaliser chaque PSD par canal
    all_psds_norm = all_psds / (all_psds.sum(axis=1, keepdims=True) + 1e-10)
    
    # Calculer le coefficient de variation pour chaque fréquence
    # Un CV faible indique que la puissance est uniforme sur tous les canaux (bruit)
    cv_per_freq = variation(all_psds_norm, axis=0)
    
    # Détecter les pics de puissance avec faible variation
    mean_power = all_psds_norm.mean(axis=0)
    power_threshold = np.percentile(mean_power, power_percentile)
    
    noise_freqs = []
    for i, freq in enumerate(freqs):
        # Bruit si: puissance élevée ET faible variation entre canaux
        if mean_power[i] > power_threshold and cv_per_freq[i] < variance_threshold:
            # Vérifier que ce n'est pas une fréquence EEG importante (< 1Hz ou 1-4Hz delta)
            if freq > 4:  # Garder les fréquences EEG basses importantes
                noise_freqs.append(freq)
    
    # Grouper les fréquences proches (±1 Hz)
    if noise_freqs:
        noise_freqs = np.array(noise_freqs)
        grouped_freqs = []
        i = 0
        while i < len(noise_freqs):
            freq_group = [noise_freqs[i]]
            j = i + 1
            while j < len(noise_freqs) and noise_freqs[j] - noise_freqs[i] < 2:
                freq_group.append(noise_freqs[j])
                j += 1
            grouped_freqs.append(np.mean(freq_group))
            i = j
        noise_freqs = grouped_freqs
    
    print(f"  ✓ Fréquences de bruit détectées: {[f'{f:.1f}Hz' for f in noise_freqs]}")
    return noise_freqs


def apply_notch_filter(raw, noise_freqs, notch_width=2):
    """
    Applique un filtre notch pour enlever les fréquences de bruit.
    
    Paramètres:
        raw: objet MNE Raw
        noise_freqs: liste des fréquences à filtrer
        notch_width: largeur du filtre notch (Hz)
    """
    if not noise_freqs:
        print("  ℹ️ Aucune fréquence de bruit à filtrer")
        return raw
    
    print(f"  🔧 Application du filtre notch sur {len(noise_freqs)} fréquences...")
    
    # Copier pour ne pas modifier l'original
    raw_filtered = raw.copy()
    
    # Appliquer le filtre notch pour chaque fréquence de bruit
    for freq in noise_freqs:
        raw_filtered.notch_filter(
            freqs=freq,
            picks='all',
            method='spectrum_fit',
            filter_length='auto',
            notch_widths=notch_width,
            verbose=False
        )
    
    print(f"  ✓ Filtrage terminé")
    return raw_filtered


def create_annotations_from_csv(df_csv, sfreq):
    """
    Crée des annotations MNE à partir du DataFrame CSV.
    
    Paramètres:
        df_csv: DataFrame des annotations CSV
        sfreq: fréquence d'échantillonnage
    
    Retourne:
        objet mne.Annotations
    """
    if df_csv is None or df_csv.empty:
        return mne.Annotations(onset=[], duration=[], description=[])
    
    onsets = []
    durations = []
    descriptions = []
    
    for _, row in df_csv.iterrows():
        onset = row['start_time']
        duration = row['stop_time'] - row['start_time']
        description = f"{row['channel']}_{row['label']}"
        
        onsets.append(onset)
        durations.append(duration)
        descriptions.append(description)
    
    annotations = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    return annotations


def process_session(csv_path: Path, edf_path: Path, outdir: Path, patient_name: str):
    """
    Traite une session complète: détection de bruit, filtrage, et sauvegarde.
    
    Paramètres:
        csv_path: chemin vers le fichier CSV d'annotations
        edf_path: chemin vers le fichier EDF
        outdir: répertoire de sortie
        patient_name: nom du patient
    """
    session_name = csv_path.stem  # e.g., aaaaadpj_s003_t000
    print(f"\n{'='*60}")
    print(f"📊 Session: {session_name}")
    print(f"{'='*60}")
    
    # Charger les annotations CSV
    df_csv = read_csv_annotations(csv_path)
    if df_csv is None:
        print("  ❌ Pas d'annotations trouvées")
        return
    
    # Charger le fichier EDF
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        print(f"  ✓ EDF chargé: {raw.n_times} échantillons, {len(raw.ch_names)} canaux")
    except Exception as e:
        print(f"  ❌ Échec de chargement EDF: {e}")
        return
    
    sfreq = raw.info['sfreq']
    
    # Détecter les fréquences de bruit
    noise_freqs = detect_noise_frequencies(raw, max_freq=60, variance_threshold=0.20, power_percentile=80)
    
    # Appliquer le filtre notch
    raw_filtered = apply_notch_filter(raw, noise_freqs, notch_width=2)
    
    # Créer les annotations MNE
    annotations = create_annotations_from_csv(df_csv, sfreq)
    raw_filtered.set_annotations(annotations)
    
    print(f"  ✓ {len(annotations)} annotations ajoutées")
    
    # Sauvegarder le fichier EDF filtré avec annotations
    outpath = outdir / f"{patient_name}_{session_name}_filtered.edf"
    try:
        # Normaliser les données pour éviter les dépassements de limites EDF
        data = raw_filtered.get_data()
        data_min = np.min(data)
        data_max = np.max(data)
        
        # Si les valeurs sont trop grandes, clipper pour rester dans les limites EDF
        if abs(data_min) > 3200 or abs(data_max) > 3200:
            print(f"  ⚠️ Données hors limites EDF (min={data_min:.1f}, max={data_max:.1f}), application de clipping...")
            data = np.clip(data, -3200, 3200)
            raw_filtered._data = data
        
        # Corriger les métadonnées problématiques dans l'info
        for ch in raw_filtered.info['chs']:
            # Limiter les valeurs physiques min/max à des valeurs raisonnables
            if 'range' in ch:
                if ch['range'] > 32767 or ch['range'] < -32768:
                    ch['range'] = 3200
            if 'cal' in ch:
                if abs(ch['cal']) > 1e6:
                    ch['cal'] = 1.0
        
        raw_filtered.export(str(outpath), overwrite=True, verbose=False)
        print(f"  ✅ Sauvegardé: {outpath.name}")
    except Exception as e:
        print(f"  ❌ Échec de sauvegarde EDF: {e}")
        # Sauvegarder en format FIF (natif MNE) comme alternative
        outpath_fif = outdir / f"{patient_name}_{session_name}_filtered.fif"
        try:
            raw_filtered.save(str(outpath_fif), overwrite=True, verbose=False)
            print(f"  ✅ Sauvegardé en format FIF: {outpath_fif.name}")
        except Exception as e2:
            print(f"  ❌ Échec complet de sauvegarde: {e2}")


def main():
    # Dossier patient
    patient_name = "aaaaaaac"
    patient_folder = Path(__file__).parent / patient_name
    
    if not patient_folder.exists():
        print(f"❌ Dossier patient non trouvé: {patient_folder}")
        return
    
    # Répertoire de sortie
    outdir = Path("filtered_edf") / f"{patient_name}_filtered"
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# Filtrage et fusion des EDF pour patient: {patient_name}")
    print(f"{'#'*60}")
    
    # Trouver tous les fichiers CSV
    csv_files = sorted(patient_folder.rglob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.endswith("_bi")]
    
    print(f"\n📁 {len(csv_files)} sessions trouvées")
    
    # Traiter chaque session
    for csv_path in csv_files:
        edf_path = csv_path.with_suffix('.edf')
        
        if not edf_path.exists():
            print(f"⚠️ Pas de fichier EDF pour {csv_path.name}, ignoré")
            continue
        
        try:
            process_session(csv_path, edf_path, outdir, patient_name)
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {csv_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'#'*60}")
    print(f"✅ Tous les fichiers EDF filtrés sauvegardés dans: {outdir.absolute()}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
