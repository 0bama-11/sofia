"""
Pipeline de entrenamiento con Transfer Learning.

Modelo: EfficientNet-B0 preentrenado en ImageNet.
Dataset: Food-101 (101 clases, ~75k train / ~25k test).
Estrategia:
  1. Congelar backbone, entrenar solo clasificador (primeras épocas).
  2. Descongelar backbone, fine-tuning completo con LR bajo.

Uso:
    python -m ml.train
    python -m ml.train --epochs 15 --batch-size 64
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import Food101

from ml.config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR,
    MODEL_NAME, NUM_CLASSES, PRETRAINED,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, LR_BACKBONE, WEIGHT_DECAY,
    FREEZE_BACKBONE_EPOCHS, UNFREEZE_AFTER,
    IMAGE_SIZE, RANDOM_CROP_SCALE, COLOR_JITTER, RANDOM_ROTATION,
    NUM_WORKERS, DEVICE, CHECKPOINT_EVERY, EARLY_STOP_PATIENCE,
)


def get_device():
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def build_transforms():
    """Data augmentation para train y normalización para val."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=RANDOM_CROP_SCALE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(RANDOM_ROTATION),
        transforms.ColorJitter(
            brightness=COLOR_JITTER,
            contrast=COLOR_JITTER,
            saturation=COLOR_JITTER,
        ),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def build_model(device):
    """Construye EfficientNet-B0 con nueva cabeza de clasificación."""
    if MODEL_NAME == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if PRETRAINED else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, NUM_CLASSES),
        )
    elif MODEL_NAME == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if PRETRAINED else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    else:
        raise ValueError(f"Modelo no soportado: {MODEL_NAME}")

    return model.to(device)


def freeze_backbone(model):
    """Congela todas las capas excepto el clasificador."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Descongela todas las capas."""
    for param in model.parameters():
        param.requires_grad = True


def build_optimizer(model, epoch):
    """Construye el optimizador con LR diferenciado."""
    if epoch < UNFREEZE_AFTER:
        # Solo entrenar clasificador
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    else:
        # Fine-tuning: LR más bajo para backbone
        classifier_params = []
        backbone_params = []
        for name, param in model.named_parameters():
            if "classifier" in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

        return torch.optim.AdamW([
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": classifier_params, "lr": LEARNING_RATE},
        ], weight_decay=WEIGHT_DECAY)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Acc: {100.*correct/total:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Top-5 accuracy
        _, top5_pred = outputs.topk(5, dim=1)
        correct_top5 += sum(
            labels[i].item() in top5_pred[i].tolist()
            for i in range(labels.size(0))
        )

    return (running_loss / total,
            100. * correct / total,
            100. * correct_top5 / total)


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
    }, path)
    print(f"  Checkpoint guardado: {path}")


def train(args):
    device = get_device()
    print(f"Dispositivo: {device}")
    print(f"Modelo: {MODEL_NAME}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    if args.resume:
        print(f"Resumiendo desde: {args.resume}")
    print()

    # ─── Datos ───
    train_transform, val_transform = build_transforms()

    print("Cargando dataset Food-101...")
    train_dataset = Food101(root=DATA_DIR, split="train",
                            transform=train_transform, download=True)
    val_dataset = Food101(root=DATA_DIR, split="test",
                          transform=val_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True)

    print(f"Train: {len(train_dataset)} imágenes")
    print(f"Val:   {len(val_dataset)} imágenes")
    print(f"Clases: {len(train_dataset.classes)}")
    print()

    # Guardar lista de clases
    os.makedirs(MODELS_DIR, exist_ok=True)
    classes_path = os.path.join(MODELS_DIR, "classes.txt")
    with open(classes_path, "w") as f:
        for cls in train_dataset.classes:
            f.write(cls + "\n")
    print(f"Clases guardadas en: {classes_path}")

    # ─── Modelo ───
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()

    # ─── Resumir desde checkpoint ───
    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0

    if args.resume:
        resume_path = args.resume
        # Si pasan "auto", buscar best_model.pt automáticamente
        if resume_path == "auto":
            resume_path = os.path.join(MODELS_DIR, "best_model.pt")

        if not os.path.exists(resume_path):
            print(f"ERROR: No se encontró checkpoint en {resume_path}")
            sys.exit(1)

        checkpoint = torch.load(resume_path, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_acc = checkpoint.get("val_acc", 0.0)
        print(f"Checkpoint cargado: época {start_epoch}, "
              f"val_acc={best_val_acc:.1f}%")
        print(f"Continuando desde época {start_epoch + 1} "
              f"hasta {start_epoch + args.epochs}")
        print()

    # ─── Entrenamiento ───
    total_epochs = start_epoch + args.epochs

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()

        # Estrategia de congelamiento
        if epoch < FREEZE_BACKBONE_EPOCHS:
            freeze_backbone(model)
            phase = "HEAD ONLY"
        else:
            if epoch == FREEZE_BACKBONE_EPOCHS or (args.resume and epoch == start_epoch and epoch >= FREEZE_BACKBONE_EPOCHS):
                unfreeze_backbone(model)
                if epoch == FREEZE_BACKBONE_EPOCHS:
                    print(f"\n>>> Backbone descongelado en época {epoch+1} <<<\n")
                else:
                    print(f"\n>>> Backbone descongelado (resume fine-tuning) <<<\n")
            phase = "FINE-TUNING"

        optimizer = build_optimizer(model, epoch)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader), eta_min=1e-6
        )

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        scheduler.step()

        # Validate
        val_loss, val_acc, val_top5 = evaluate(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - t0
        print(f"\nÉpoca {epoch+1}/{total_epochs} [{phase}] "
              f"({elapsed:.0f}s)")
        print(f"  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
        print(f"  Val   — Loss: {val_loss:.4f} | "
              f"Top-1: {val_acc:.1f}% | Top-5: {val_top5:.1f}%")

        # Checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            cp_path = os.path.join(MODELS_DIR, f"checkpoint_epoch{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, val_acc, cp_path)

        # Mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_path = os.path.join(MODELS_DIR, "best_model.pt")
            save_checkpoint(model, optimizer, epoch, val_acc, best_path)
            print(f"  ★ Nuevo mejor modelo: {val_acc:.1f}%")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping: sin mejora en {EARLY_STOP_PATIENCE} "
                  f"épocas.")
            break

        print()

    print(f"\nEntrenamiento completado. Mejor Val Top-1: {best_val_acc:.1f}%")
    print(f"Modelo guardado en: {os.path.join(MODELS_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo Food-101")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Número de épocas adicionales a entrenar")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--resume", type=str, default=None,
                        help='Ruta al checkpoint para continuar entrenamiento. '
                             'Usa "auto" para cargar best_model.pt automáticamente.')
    args = parser.parse_args()
    train(args)
