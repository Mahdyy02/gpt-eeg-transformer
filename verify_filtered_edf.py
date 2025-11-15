from pathlib import Path
import mne
import pandas as pd

"""
verify_filtered_edf.py

Vérifie les fichiers EDF filtrés et affiche leurs informations et annotations.
"""

def verify_file(file_path: Path):
    """Vérifie et affiche les informations d'un fichier filtré."""
    print(f"\n{'='*60}")
    print(f"📄 Fichier: {file_path.name}")
    print(f"{'='*60}")
    
    try:
        # Charger le fichier
        if file_path.suffix == '.fif':
            raw = mne.io.read_raw_fif(str(file_path), preload=False, verbose=False)
        else:
            raw = mne.io.read_raw_edf(str(file_path), preload=False, verbose=False)
        
        # Informations générales
        sfreq = raw.info['sfreq']
        duration = raw.n_times / sfreq
        n_channels = len(raw.ch_names)
        
        print(f"  ✓ Format: {file_path.suffix}")
        print(f"  ✓ Fréquence d'échantillonnage: {sfreq} Hz")
        print(f"  ✓ Durée: {duration:.2f} secondes ({duration/60:.2f} minutes)")
        print(f"  ✓ Nombre de canaux: {n_channels}")
        print(f"  ✓ Nombre d'échantillons: {raw.n_times}")
        
        # Lister les premiers canaux
        print(f"\n  📡 Canaux (premiers 10):")
        for i, ch in enumerate(raw.ch_names[:10]):
            print(f"      {i+1}. {ch}")
        if n_channels > 10:
            print(f"      ... et {n_channels - 10} autres")
        
        # Annotations
        annotations = raw.annotations
        print(f"\n  📝 Annotations: {len(annotations)} au total")
        
        if len(annotations) > 0:
            # Créer un DataFrame pour analyser les annotations
            df_annot = pd.DataFrame({
                'onset': annotations.onset,
                'duration': annotations.duration,
                'description': annotations.description
            })
            
            # Extraire le canal et le label de la description
            df_annot['channel'] = df_annot['description'].str.split('_').str[0]
            df_annot['label'] = df_annot['description'].str.split('_').str[1]
            
            # Compter par label
            label_counts = df_annot['label'].value_counts()
            print(f"\n  📊 Répartition par type:")
            for label, count in label_counts.items():
                total_duration = df_annot[df_annot['label'] == label]['duration'].sum()
                print(f"      • {label}: {count} segments ({total_duration:.1f}s)")
            
            # Afficher quelques exemples
            print(f"\n  🔍 Exemples d'annotations (5 premiers):")
            for i, row in df_annot.head(5).iterrows():
                print(f"      {row['channel']:15} | {row['onset']:8.2f}s - {row['onset']+row['duration']:8.2f}s | {row['label']}")
        
        print(f"\n  ✅ Fichier valide et lisible")
        
    except Exception as e:
        print(f"  ❌ Erreur lors de la lecture: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Dossier des fichiers filtrés
    filtered_dir = Path(__file__).parent / "filtered_edf" / "aaaaadpj"
    
    if not filtered_dir.exists():
        print(f"❌ Dossier non trouvé: {filtered_dir}")
        return
    
    print(f"\n{'#'*60}")
    print(f"# Vérification des fichiers EDF filtrés")
    print(f"# Dossier: {filtered_dir}")
    print(f"{'#'*60}")
    
    # Trouver tous les fichiers EDF et FIF
    files = sorted(list(filtered_dir.glob("*.edf")) + list(filtered_dir.glob("*.fif")))
    
    print(f"\n📁 {len(files)} fichiers trouvés")
    
    # Vérifier chaque fichier
    for file_path in files:
        verify_file(file_path)
    
    print(f"\n{'#'*60}")
    print(f"✅ Vérification terminée")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
