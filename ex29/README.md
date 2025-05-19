# Exercice 29 : Analyse Géospatiale avec Imagerie Multi-Spectrale

## Objectif
Cet exercice vise à développer un pipeline complet d'analyse géospatiale en utilisant des données multi-spectrales (Sentinel-2 et Sentinel-1). Le but est de prétraiter, fusionner et classifier les données pour identifier différentes classes d'utilisation des terres.

## Installation

1. Créez un environnement virtuel Python :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Structure du Projet

- `main.py` : Pipeline principal (prétraitement, fusion, classification)
- `download_data.py` : Téléchargement des données Sentinel
- `visualize.py` : Visualisation des résultats
- `create_test_data.py` : Génération de données de test
- `config.yaml` : Configuration du projet
- `requirements.txt` : Dépendances Python

## Utilisation

### Génération de Données de Test
```bash
python create_test_data.py
```

### Exécution du Pipeline
```bash
python main.py --input_dir input --output_dir output
```

### Visualisation des Résultats
```bash
python visualize.py --input_dir output --output_dir output --config config.yaml
```

## Résultats

Le pipeline génère :
- `classification.npy` : Résultats de la classification
- `classification_report.txt` : Rapport de performance
- `classification_map.png` : Carte de classification
- `confusion_matrix.png` : Matrice de confusion
- `temporal_analysis.png` : Analyse temporelle

## Configuration

Modifiez `config.yaml` pour ajuster :
- Paramètres d'accès aux données Sentinel
- Région d'étude
- Dates d'analyse
- Paramètres de classification
- Options de visualisation