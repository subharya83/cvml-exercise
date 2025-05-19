#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import yaml

def load_config(config_path):
    """Charge la configuration depuis un fichier YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_classification_map(classification, config, output_path):
    """Génère une carte de classification."""
    plt.figure(figsize=tuple(config['visualization']['figure_size']))
    plt.imshow(classification, cmap=config['visualization']['colormap'])
    plt.colorbar(label='Classe')
    plt.title('Carte de Classification')
    plt.savefig(output_path, dpi=config['visualization']['dpi'])
    plt.close()

def plot_confusion_matrix(y_true, y_pred, config, output_path):
    """Génère une matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=tuple(config['visualization']['figure_size']))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité')
    plt.savefig(output_path, dpi=config['visualization']['dpi'])
    plt.close()

def plot_temporal_analysis(data, config, output_path):
    """Génère une analyse temporelle."""
    plt.figure(figsize=tuple(config['visualization']['figure_size']))
    for i, class_name in enumerate(config['classification']['classes']):
        plt.plot(data[:, i], label=class_name)
    plt.title('Analyse Temporelle des Classes')
    plt.xlabel('Temps')
    plt.ylabel('Proportion')
    plt.legend()
    plt.savefig(output_path, dpi=config['visualization']['dpi'])
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualisation des résultats')
    parser.add_argument('--input_dir', type=str, required=True, help='Répertoire des données d\'entrée')
    parser.add_argument('--output_dir', type=str, required=True, help='Répertoire de sortie')
    parser.add_argument('--config', type=str, default='config.yaml', help='Fichier de configuration')
    args = parser.parse_args()

    # Création du répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)

    # Chargement de la configuration
    config = load_config(args.config)

    # Chargement des données
    classification = np.load(os.path.join(args.input_dir, 'classification.npy'))
    
    if os.path.exists(os.path.join(args.input_dir, 'labels.npy')):
        labels = np.load(os.path.join(args.input_dir, 'labels.npy'))
        plot_confusion_matrix(labels, classification, config,
                            os.path.join(args.output_dir, 'confusion_matrix.png'))

    # Génération des visualisations
    plot_classification_map(classification, config,
                          os.path.join(args.output_dir, 'classification_map.png'))
    
    if os.path.exists(os.path.join(args.input_dir, 'temporal_data.npy')):
        temporal_data = np.load(os.path.join(args.input_dir, 'temporal_data.npy'))
        plot_temporal_analysis(temporal_data, config,
                             os.path.join(args.output_dir, 'temporal_analysis.png'))

    print("Visualisation terminée !")

if __name__ == '__main__':
    main() 