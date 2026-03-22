"""
Configuración central del pipeline de entrenamiento.
Modificar estos valores para ajustar el entrenamiento.
"""

import os

# ─── Rutas ───
ML_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ML_DIR)
DATA_DIR = os.path.join(ML_DIR, "data")
MODELS_DIR = os.path.join(ML_DIR, "models")
LOGS_DIR = os.path.join(ML_DIR, "logs")

# ─── Modelo ───
MODEL_NAME = "efficientnet_b0"  # Buen balance entre precisión y tamaño
NUM_CLASSES = 101               # Food-101 tiene 101 clases
PRETRAINED = True

# ─── Entrenamiento ───
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3       # Para la cabeza nueva
LR_BACKBONE = 1e-4         # Para fine-tuning del backbone (más bajo)
WEIGHT_DECAY = 1e-4

# ─── Transfer learning ───
FREEZE_BACKBONE_EPOCHS = 3  # Entrenar solo la cabeza las primeras N épocas
UNFREEZE_AFTER = 3          # Después de esta época, hacer fine-tuning completo

# ─── Data augmentation ───
IMAGE_SIZE = 224            # EfficientNet-B0 espera 224x224
RANDOM_CROP_SCALE = (0.08, 1.0)
COLOR_JITTER = 0.3
RANDOM_ROTATION = 15

# ─── Otros ───
NUM_WORKERS = 2             # Workers para DataLoader
DEVICE = "auto"             # "auto", "cuda", "cpu"
CHECKPOINT_EVERY = 2        # Guardar checkpoint cada N épocas
EARLY_STOP_PATIENCE = 4    # Parar si no mejora en N épocas
