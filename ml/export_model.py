"""
Exportar el modelo entrenado a formato optimizado para inferencia.

Genera:
  1. model_inference.pt  — Solo pesos (state_dict), liviano
  2. model_scripted.pt   — TorchScript, para producción
  3. metadata.json       — Info del modelo + lista de clases

Uso:
    python -m ml.export_model
    python -m ml.export_model --model ml/models/best_model.pt
"""

import os
import json
import argparse

import torch

from ml.config import MODELS_DIR, IMAGE_SIZE, NUM_CLASSES, MODEL_NAME
from ml.train import build_model, get_device
from ml.evaluate import load_classes


EXPORT_DIR = os.path.join(MODELS_DIR, "export")


def export_model(args):
    device = get_device()
    model_path = args.model or os.path.join(MODELS_DIR, "best_model.pt")

    if not os.path.exists(model_path):
        print(f"Error: No se encontró {model_path}")
        return

    print(f"Cargando modelo desde: {model_path}")
    model = build_model(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_acc = checkpoint.get("val_acc", 0)
    epoch = checkpoint.get("epoch", "?")
    print(f"  Época: {epoch}, Val Acc: {val_acc:.1f}%")

    os.makedirs(EXPORT_DIR, exist_ok=True)

    # ─── 1. State dict (liviano) ───
    inference_path = os.path.join(EXPORT_DIR, "model_inference.pt")
    torch.save(model.state_dict(), inference_path)
    size_mb = os.path.getsize(inference_path) / (1024 * 1024)
    print(f"  State dict: {inference_path} ({size_mb:.1f} MB)")

    # ─── 2. TorchScript ───
    scripted_path = os.path.join(EXPORT_DIR, "model_scripted.pt")
    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    scripted_model = torch.jit.trace(model, example_input)
    scripted_model.save(scripted_path)
    size_mb = os.path.getsize(scripted_path) / (1024 * 1024)
    print(f"  TorchScript: {scripted_path} ({size_mb:.1f} MB)")

    # ─── 3. Metadata ───
    classes = load_classes()
    metadata = {
        "model_name": MODEL_NAME,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "val_accuracy_top1": round(val_acc, 2),
        "epoch": epoch,
        "classes": classes,
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    meta_path = os.path.join(EXPORT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {meta_path}")

    print(f"\nExportación completa en: {EXPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    export_model(args)
