#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
from datetime import datetime
from sentinelsat import SentinelAPI
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_config(config_path):
    """Charge la configuration depuis un fichier YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_sentinel_data(config, output_dir):
    """Télécharge les données Sentinel."""
    # Connexion à l'API Sentinel
    api = SentinelAPI(
        config['sentinel']['username'],
        config['sentinel']['password'],
        config['sentinel']['api_url']
    )

    # Paramètres de recherche
    bbox = config['processing']['region']['bbox']
    start_date = config['processing']['dates']['start']
    end_date = config['processing']['dates']['end']

    # Recherche des produits Sentinel-2
    products_s2 = api.query(
        bbox,
        date=(start_date, end_date),
        platformname='Sentinel-2',
        cloudcoverpercentage=(0, 20)
    )

    # Recherche des produits Sentinel-1
    products_s1 = api.query(
        bbox,
        date=(start_date, end_date),
        platformname='Sentinel-1',
        producttype='GRD'
    )

    # Téléchargement des données
    print("Téléchargement des données Sentinel-2...")
    api.download_all(products_s2, output_dir)

    print("Téléchargement des données Sentinel-1...")
    api.download_all(products_s1, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Téléchargement des données Sentinel')
    parser.add_argument('--config', type=str, default='config.yaml', help='Fichier de configuration')
    parser.add_argument('--output_dir', type=str, default='input', help='Répertoire de sortie')
    args = parser.parse_args()

    # Création du répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)

    # Chargement de la configuration
    config = load_config(args.config)

    # Téléchargement des données
    download_sentinel_data(config, args.output_dir)

    print("Téléchargement terminé !")

if __name__ == '__main__':
    main() 