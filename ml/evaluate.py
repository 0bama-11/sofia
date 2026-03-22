"""
Evaluación detallada del modelo entrenado.

Genera:
  - Top-1 y Top-5 accuracy
  - Precision, recall, F1 por clase
  - Matriz de confusión (top errores)
  - Reporte resumen

Uso:
    python -m ml.evaluate
    python -m ml.evaluate --model ml/models/best_model.pt
"""

import os
import argparse
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Food101

from ml.config import (
    DATA_DIR, MODELS_DIR, IMAGE_SIZE, NUM_WORKERS, BATCH_SIZE, NUM_CLASSES,
)
from ml.train import build_model, get_device


def load_classes(models_dir=MODELS_DIR):
    classes_path = os.path.join(models_dir, "classes.txt")
    with open(classes_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_trained_model(model_path, device):
    model = build_model(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_acc", 0)
    print(f"Modelo cargado: época {epoch}, val_acc={val_acc:.1f}%")
    return model


@torch.no_grad()
def full_evaluation(model, loader, device, classes):
    """Evaluación completa con métricas por clase."""
    # Contadores por clase
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    total = 0
    correct_top1 = 0
    correct_top5 = 0

    # Pares de confusión más frecuentes
    confusion_pairs = defaultdict(int)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        # Top-1
        _, pred = outputs.max(1)
        correct_top1 += pred.eq(labels).sum().item()

        # Top-5
        _, top5 = outputs.topk(5, dim=1)
        for i in range(labels.size(0)):
            label = labels[i].item()
            predicted = pred[i].item()
            total += 1
            class_total[label] += 1

            if label in top5[i].tolist():
                correct_top5 += 1

            if predicted == label:
                class_correct[label] += 1
                class_tp[label] += 1
            else:
                class_fp[predicted] += 1
                class_fn[label] += 1
                pair_key = (classes[label], classes[predicted])
                confusion_pairs[pair_key] += 1

    # ─── Reporte global ───
    top1_acc = 100. * correct_top1 / total
    top5_acc = 100. * correct_top5 / total
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN COMPLETA — {total} imágenes")
    print(f"{'='*60}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    # ─── Métricas por clase ───
    print(f"\n{'─'*60}")
    print(f"{'Clase':<30} {'Prec':>6} {'Recall':>7} {'F1':>6} {'N':>5}")
    print(f"{'─'*60}")

    f1_scores = []
    for idx in range(len(classes)):
        tp = class_tp[idx]
        fp = class_fp[idx]
        fn = class_fn[idx]
        n = class_total[idx]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)
        f1_scores.append((classes[idx], f1, precision, recall, n))

    # Ordenar por F1 descendente
    f1_scores.sort(key=lambda x: x[1], reverse=True)

    for name, f1, prec, rec, n in f1_scores:
        print(f"{name:<30} {prec:>5.1%} {rec:>6.1%} {f1:>5.1%} {n:>5}")

    # ─── Top confusiones ───
    print(f"\n{'─'*60}")
    print("TOP 15 CONFUSIONES (real → predicho):")
    print(f"{'─'*60}")
    sorted_conf = sorted(confusion_pairs.items(),
                         key=lambda x: x[1], reverse=True)[:15]
    for (real, pred), count in sorted_conf:
        print(f"  {real:<25} → {pred:<25} ({count} veces)")

    # ─── Peores clases ───
    print(f"\n{'─'*60}")
    print("10 PEORES CLASES (menor F1):")
    print(f"{'─'*60}")
    for name, f1, prec, rec, n in f1_scores[-10:]:
        print(f"  {name:<30} F1={f1:.1%}  Prec={prec:.1%}  Rec={rec:.1%}")

    return top1_acc, top5_acc


def main(args):
    device = get_device()
    classes = load_classes()

    model_path = args.model or os.path.join(MODELS_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        print("Primero entrena con: python -m ml.train")
        return

    model = load_trained_model(model_path, device)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = Food101(root=DATA_DIR, split="test",
                          transform=val_transform, download=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    full_evaluation(model, val_loader, device, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Ruta al .pt del modelo")
    args = parser.parse_args()
    main(args)
