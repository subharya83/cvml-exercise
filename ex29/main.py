#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import numpy as np
import rasterio
from rasterio.plot import show
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_config(config_path):
    """Charge la configuration depuis un fichier YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_sentinel2(image_path):
    """Prétraite les données Sentinel-2."""
    with rasterio.open(image_path) as src:
        image = src.read()
        # Normalisation des bandes
        image = image / 10000.0
        # Masquage des nuages (simplifié)
        mask = image[0] > 0.3
        image = np.where(mask[None, :, :], image, 0)
    return image

def preprocess_sentinel1(image_path):
    """Prétraite les données Sentinel-1."""
    with rasterio.open(image_path) as src:
        image = src.read()
        # Filtrage du speckle (simplifié)
        image = np.where(image > 0, image, 0)
    return image

def fuse_sensors(optical_data, radar_data):
    """Fusionne les données optiques et radar."""
    # Alignement temporel (simplifié)
    if optical_data.shape[1:] != radar_data.shape[1:]:
        radar_data = np.resize(radar_data, optical_data.shape)
    
    # Fusion au niveau des pixels
    fused_data = np.concatenate([optical_data, radar_data], axis=0)
    return fused_data

def extract_features(data):
    """Extrait les caractéristiques des données fusionnées."""
    n_bands = data.shape[0]
    n_samples = data.shape[1] * data.shape[2]
    
    # Réorganisation des données pour l'apprentissage
    features = data.reshape(n_bands, n_samples).T
    return features

def classify_land_use(features, labels=None, train=True):
    """Classifie l'utilisation des terres."""
    if train and labels is not None:
        # Aplatir les labels pour correspondre aux features
        labels_flat = labels.reshape(-1)
        # Entraînement du modèle
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(features, labels_flat)
        return clf
    else:
        # Prédiction
        predictions = clf.predict(features)
        # Remettre les prédictions dans la forme originale
        return predictions.reshape(labels.shape)

def main():
    parser = argparse.ArgumentParser(description='Analyse géospatiale avec imagerie multi-spectrale')
    parser.add_argument('--input_dir', type=str, required=True, help='Répertoire des données d\'entrée')
    parser.add_argument('--output_dir', type=str, required=True, help='Répertoire de sortie')
    parser.add_argument('--config', type=str, default='config.yaml', help='Fichier de configuration')
    args = parser.parse_args()

    # Création des répertoires de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'assets'), exist_ok=True)

    # Chargement de la configuration
    config = load_config(args.config)

    # Traitement des données
    print("Prétraitement des données Sentinel-2...")
    optical_data = preprocess_sentinel2(os.path.join(args.input_dir, 'sentinel2.tif'))
    
    print("Prétraitement des données Sentinel-1...")
    radar_data = preprocess_sentinel1(os.path.join(args.input_dir, 'sentinel1.tif'))
    
    print("Fusion des capteurs...")
    fused_data = fuse_sensors(optical_data, radar_data)
    
    print("Extraction des caractéristiques...")
    features = extract_features(fused_data)
    
    print("Classification...")
    if os.path.exists(os.path.join(args.input_dir, 'labels.npy')):
        labels = np.load(os.path.join(args.input_dir, 'labels.npy'))
        clf = classify_land_use(features, labels, train=True)
        predictions = clf.predict(features).reshape(labels.shape)
    else:
        predictions = classify_land_use(features, train=False)
    
    # Sauvegarde des résultats
    np.save(os.path.join(args.output_dir, 'classification.npy'), predictions)
    
    # Génération du rapport de classification
    if labels is not None:
        report = classification_report(labels.reshape(-1), predictions.reshape(-1))
        with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
    
    print("Traitement terminé !")

if __name__ == '__main__':
    main() 