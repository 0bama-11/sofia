"""
Descarga y preparación del dataset Food-101.

Food-101 contiene 101 categorías de comida con 1000 imágenes cada una.
Usamos torchvision.datasets para descarga automática.

Uso:
    python -m ml.download_dataset
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "ml", "data")


def download_food101():
    """Descarga Food-101 usando torchvision."""
    from torchvision.datasets import Food101

    print(f"Directorio de datos: {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Descargando Food-101 (split train)...")
    train_ds = Food101(root=DATA_DIR, split="train", download=True)
    print(f"  Train: {len(train_ds)} imágenes")

    print("Descargando Food-101 (split test)...")
    test_ds = Food101(root=DATA_DIR, split="test", download=True)
    print(f"  Test:  {len(test_ds)} imágenes")

    # Mostrar las clases
    classes = train_ds.classes
    print(f"\nTotal de clases: {len(classes)}")
    print("Primeras 20 clases:")
    for i, c in enumerate(classes[:20]):
        print(f"  {i:3d}. {c}")

    return classes


if __name__ == "__main__":
    download_food101()
