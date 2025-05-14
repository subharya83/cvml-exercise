#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_test_data(output_dir):
    """Crée des données de test pour l'analyse géospatiale."""
    # Création du répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Paramètres de l'image
    height = 100
    width = 100
    n_bands = 13  # Nombre de bandes Sentinel-2

    # Création des données Sentinel-2
    sentinel2_data = np.random.rand(n_bands, height, width) * 10000
    transform = from_origin(2.2241, 48.9022, 0.0001, 0.0001)
    
    with rasterio.open(
        os.path.join(output_dir, 'sentinel2.tif'),
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=n_bands,
        dtype=np.float32,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(sentinel2_data)

    # Création des données Sentinel-1
    sentinel1_data = np.random.rand(2, height, width) * 1000
    with rasterio.open(
        os.path.join(output_dir, 'sentinel1.tif'),
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=2,
        dtype=np.float32,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(sentinel1_data)

    # Création des labels
    labels = np.random.randint(0, 4, size=(height, width))
    np.save(os.path.join(output_dir, 'labels.npy'), labels)

    # Création des données temporelles
    temporal_data = np.random.rand(12, 4)  # 12 mois, 4 classes
    np.save(os.path.join(output_dir, 'temporal_data.npy'), temporal_data)

if __name__ == '__main__':
    create_test_data('input') 