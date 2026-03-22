"""
Servicio de inferencia.

Carga el modelo entrenado (TorchScript o state_dict) y ejecuta
predicción real sobre imágenes. Si el modelo no está disponible,
cae en modo simulado como fallback.
"""

import os
import json
import random
import logging

from PIL import Image

logger = logging.getLogger(__name__)

# ─── Rutas del modelo exportado ───
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXPORT_DIR = os.path.join(_ROOT, "ml", "models", "export")
_SCRIPTED_PATH = os.path.join(_EXPORT_DIR, "model_scripted.pt")
_META_PATH = os.path.join(_EXPORT_DIR, "metadata.json")

# Estado global del modelo (singleton lazy)
_model = None
_classes = None
_transform = None
_device = None
_model_loaded = False
_load_attempted = False


def _try_load_model():
    """Intenta cargar el modelo una sola vez."""
    global _model, _classes, _transform, _device, _model_loaded, _load_attempted

    if _load_attempted:
        return _model_loaded
    _load_attempted = True

    try:
        import torch
        from torchvision import transforms

        if not os.path.exists(_SCRIPTED_PATH):
            logger.info("Modelo no encontrado en %s — modo simulado", _SCRIPTED_PATH)
            return False

        # Cargar metadata
        with open(_META_PATH, "r") as f:
            meta = json.load(f)

        _classes = meta["classes"]
        img_size = meta.get("image_size", 224)
        norm_mean = meta["normalize"]["mean"]
        norm_std = meta["normalize"]["std"]

        # Transform de inferencia
        _transform = transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

        # Cargar modelo
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = torch.jit.load(_SCRIPTED_PATH, map_location=_device)
        _model.eval()

        _model_loaded = True
        logger.info("Modelo cargado correctamente (%d clases, device=%s)",
                     len(_classes), _device)
        return True

    except Exception as e:
        logger.warning("Error cargando modelo: %s — modo simulado", e)
        return False


def predict_food(image_path: str) -> list[dict]:
    """
    Predice la clase de comida en la imagen.
    Devuelve top-3 clases con confianza.

    Si el modelo está entrenado y exportado, usa inferencia real.
    Si no, cae en modo simulado (aleatorio).
    """
    if _try_load_model():
        return _predict_real(image_path)
    else:
        return _predict_simulated()


def _predict_real(image_path: str) -> list[dict]:
    """Inferencia real con el modelo entrenado."""
    import torch

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error("No se pudo abrir la imagen: %s", e)
        return _predict_simulated()

    input_tensor = _transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        output = _model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Top-3
    top3_prob, top3_idx = torch.topk(probabilities, 3)

    # Convertir a nombres con mapeo español
    from ml.class_mapping import food101_to_spanish, food101_to_food_table_key

    results = []
    for prob, idx in zip(top3_prob, top3_idx):
        class_name = _classes[idx.item()]
        food_table_key = food101_to_food_table_key(class_name)
        spanish_name = food101_to_spanish(class_name)

        results.append({
            "food_class": food_table_key or class_name.replace("_", " "),
            "display_name": spanish_name,
            "original_class": class_name,
            "confidence": round(prob.item(), 3),
        })

    return results


def _predict_simulated() -> list[dict]:
    """Fallback: predicción aleatoria cuando no hay modelo."""
    from app.nutrition.food_table import get_food_classes

    classes = get_food_classes()
    top3 = random.sample(classes, min(3, len(classes)))

    confidences = sorted(
        [random.uniform(0.3, 0.95) for _ in top3], reverse=True
    )

    results = []
    for cls, conf in zip(top3, confidences):
        results.append({
            "food_class": cls,
            "display_name": cls.title(),
            "original_class": cls,
            "confidence": round(conf, 2),
        })

    return results


def is_model_loaded() -> bool:
    """Devuelve si el modelo real está disponible."""
    _try_load_model()
    return _model_loaded
