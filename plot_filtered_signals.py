from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd

"""
plot_filtered_signals.py

Plot les signaux EEG filtrés avec les annotations pre-ictal/ictal/post-ictal.
"""

def get_bipolar_signal(raw, ch_bipolar: str):
    """
    Crée un signal bipolaire (e.g., 'FP1-F7') à partir de deux canaux monopolaires.
    Retourne (data, times) ou None si les électrodes sont manquantes.
    """
    ch_bipolar = ch_bipolar.strip().upper()
    if '-' not in ch_bipolar:
        return None

    ch1, ch2 = ch_bipolar.split('-')
    ch1, ch2 = ch1.strip(), ch2.strip()

    # Helper pour matcher les noms de canaux monopolaires
    def match_channel(name):
        for c in raw.ch_names:
            c_clean = c.upper().replace('EEG', '').replace('_', '').replace(' ', '').replace('-REF', '').replace('-LE', '')
            if name in c_clean:
                return c
        return None

    ch1_full = match_channel(ch1)
    ch2_full = match_channel(ch2)

    if not ch1_full or not ch2_full:
        print(f"  ⚠️ Électrodes manquantes pour {ch_bipolar}: {ch1_full}, {ch2_full}")
        return None

    data = raw.get_data(picks=[ch1_full])[0] - raw.get_data(picks=[ch2_full])[0]
    return data, raw.times


def plot_filtered_session(file_path: Path, outdir: Path, channels_to_plot=None):
    """
    Plot les canaux d'une session filtrée avec les annotations.
    
    Paramètres:
        file_path: chemin vers le fichier filtré (.edf ou .fif)
        outdir: répertoire de sortie pour les plots
        channels_to_plot: liste des canaux bipolaires à plotter (None = tous)
    """
    session_name = file_path.stem.replace('_filtered', '')
    print(f"\n{'='*60}")
    print(f"📊 Session: {session_name}")
    print(f"{'='*60}")
    
    # Charger le fichier
    try:
        if file_path.suffix == '.fif':
            raw = mne.io.read_raw_fif(str(file_path), preload=True, verbose=False)
        else:
            raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
    except Exception as e:
        print(f"  ❌ Échec de chargement: {e}")
        return
    
    times = raw.times
    annotations = raw.annotations
    
    # Extraire les canaux uniques des annotations
    if len(annotations) == 0:
        print(f"  ⚠️ Aucune annotation trouvée")
        return
    
    # Créer un DataFrame des annotations
    df_annot = pd.DataFrame({
        'onset': annotations.onset,
        'duration': annotations.duration,
        'description': annotations.description
    })
    
    # Extraire le canal et le label
    df_annot['channel'] = df_annot['description'].str.split('_').str[0]
    df_annot['label'] = df_annot['description'].str.split('_').str[1]
    
    # Canaux uniques
    unique_channels = df_annot['channel'].unique()
    
    if channels_to_plot is None:
        channels_to_plot = unique_channels
    else:
        # Filtrer les canaux demandés qui existent dans les annotations
        channels_to_plot = [ch for ch in channels_to_plot if ch in unique_channels]
    
    print(f"  📡 Canaux à plotter: {len(channels_to_plot)}")
    
    # Couleurs pour les phases
    colors = {
        'bckg': {'pre': '#FF8C00', 'post': '#1E90FF'},
        'cpsz': '#DC143C', 'fnsz': '#DC143C', 'gnsz': '#DC143C', 
        'tcsz': '#DC143C', 'spsz': '#DC143C', 'absz': '#DC143C',
    }
    
    # Plotter chaque canal
    for channel_name in channels_to_plot:
        # Obtenir les annotations pour ce canal
        channel_annotations = df_annot[df_annot['channel'] == channel_name].copy().sort_values('onset')
        
        if channel_annotations.empty:
            continue
        
        # Reconstruire le signal bipolaire
        result = get_bipolar_signal(raw, channel_name)
        if result is None:
            print(f"  ⚠️ Impossible de reconstruire {channel_name}, ignoré")
            continue
        
        data, signal_times = result
        
        # Créer le plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 5))
        
        seizure_occurred = False
        legend_labels = set()
        
        for idx, row in channel_annotations.iterrows():
            start_time = row['onset']
            stop_time = row['onset'] + row['duration']
            label = row['label']
            
            mask = (signal_times >= start_time) & (signal_times <= stop_time)
            segment_times = signal_times[mask]
            segment_data = data[mask]
            
            if len(segment_times) == 0:
                continue
            
            # Déterminer la couleur et le label de phase
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
            
            # Plotter le segment
            if phase_label not in legend_labels:
                ax.plot(segment_times, segment_data, color=color, linewidth=0.7, label=phase_label)
                legend_labels.add(phase_label)
            else:
                ax.plot(segment_times, segment_data, color=color, linewidth=0.7)
        
        # Configuration du plot
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(f'{channel_name} (μV)', fontsize=12)
        ax.set_title(f"{session_name} — {channel_name} [FILTERED]", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Légende
        handles, labels_list = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        # Sauvegarder
        safe_channel_name = channel_name.replace(' ', '_').replace('/', '-')
        outpath = outdir / f"{session_name}_filtered_{safe_channel_name}.png"
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✅ Sauvegardé: {safe_channel_name}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot les signaux EEG filtrés avec annotations'
    )
    parser.add_argument(
        '--channels',
        nargs='*',
        default=None,
        help='Canaux bipolaires à plotter (ex: FP1-F7 F7-T3). Si non spécifié, tous les canaux sont plotés.'
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Plotter seulement un échantillon de canaux (premiers 5)'
    )
    args = parser.parse_args()
    
    # Dossier des fichiers filtrés
    filtered_dir = Path(__file__).parent / "filtered_edf" / "aaaaadpj"
    
    if not filtered_dir.exists():
        print(f"❌ Dossier non trouvé: {filtered_dir}")
        return
    
    # Répertoire de sortie
    outdir = Path("plots") / "aaaaadpj_filtered"
    outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# Plot des signaux EEG filtrés")
    print(f"# Patient: aaaaadpj")
    print(f"{'#'*60}")
    
    # Trouver tous les fichiers
    files = sorted(list(filtered_dir.glob("*.edf")) + list(filtered_dir.glob("*.fif")))
    
    print(f"\n📁 {len(files)} fichiers à traiter")
    
    # Déterminer les canaux à plotter
    channels_to_plot = args.channels
    if args.sample:
        # Canaux d'exemple
        channels_to_plot = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'T3-C3']
        print(f"  📊 Mode échantillon: {len(channels_to_plot)} canaux")
    elif channels_to_plot:
        print(f"  📊 Canaux spécifiés: {channels_to_plot}")
    else:
        print(f"  📊 Tous les canaux seront plotés")
    
    # Traiter chaque fichier
    for file_path in files:
        try:
            plot_filtered_session(file_path, outdir, channels_to_plot)
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'#'*60}")
    print(f"✅ Plots sauvegardés dans: {outdir.absolute()}")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
