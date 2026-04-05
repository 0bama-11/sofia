"""
Servicio de inferencia — v2 con Claude Vision.

Jerarquía de motores:
  1. Modelo PyTorch local  → cuando esté entrenado y exportado (futuro)
  2. Claude Vision API     → motor principal actual (requiere ANTHROPIC_API_KEY)
  3. Fallback simulado     → solo en desarrollo sin internet ni API key

Configura tu API key en una variable de entorno:
    export ANTHROPIC_API_KEY="sk-ant-..."
O crea un archivo .env en la raíz del proyecto:
    ANTHROPIC_API_KEY=sk-ant-...
"""

import os
import json
import base64
import logging
import random

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# 1. CONFIG — API KEY
# ──────────────────────────────────────────
def _get_api_key() -> str | None:
    """Busca la API key en env o en archivo .env de la raíz."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(root, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ──────────────────────────────────────────
# 2. MODELO PYTORCH (futuro — sin cambios)
# ──────────────────────────────────────────
_ROOT          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXPORT_DIR    = os.path.join(_ROOT, "ml", "models", "export")
_SCRIPTED_PATH = os.path.join(_EXPORT_DIR, "model_scripted.pt")
_META_PATH     = os.path.join(_EXPORT_DIR, "metadata.json")

_model          = None
_classes        = None
_transform      = None
_device         = None
_model_loaded   = False
_load_attempted = False


def _try_load_model() -> bool:
    global _model, _classes, _transform, _device, _model_loaded, _load_attempted
    if _load_attempted:
        return _model_loaded
    _load_attempted = True
    try:
        import torch
        from torchvision import transforms
        if not os.path.exists(_SCRIPTED_PATH):
            logger.info("Modelo PyTorch no encontrado — usando Claude Vision")
            return False
        with open(_META_PATH, "r") as f:
            meta = json.load(f)
        _classes   = meta["classes"]
        img_size   = meta.get("image_size", 224)
        norm_mean  = meta["normalize"]["mean"]
        norm_std   = meta["normalize"]["std"]
        _transform = transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model  = torch.jit.load(_SCRIPTED_PATH, map_location=_device)
        _model.eval()
        _model_loaded = True
        logger.info("Modelo PyTorch cargado (%d clases, device=%s)", len(_classes), _device)
        return True
    except Exception as e:
        logger.warning("Error cargando modelo PyTorch: %s — usando Claude Vision", e)
        return False


def _predict_pytorch(image_path: str) -> list[dict]:
    import torch
    from PIL import Image
    from ml.class_mapping import food101_to_spanish, food101_to_food_table_key
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error("No se pudo abrir la imagen: %s", e)
        return []
    input_tensor = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        output = _model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    results = []
    for prob, idx in zip(top3_prob, top3_idx):
        class_name     = _classes[idx.item()]
        food_table_key = food101_to_food_table_key(class_name)
        spanish_name   = food101_to_spanish(class_name)
        results.append({
            "food_class":     food_table_key or class_name.replace("_", " "),
            "display_name":   spanish_name,
            "original_class": class_name,
            "confidence":     round(prob.item(), 3),
            "source":         "pytorch",
            "description":    "",
            "calories":       None,
            "protein":        None,
            "carbs":          None,
            "fat":            None,
            "fiber":          None,
        })
    return results


# ──────────────────────────────────────────
# 3. CLAUDE VISION — motor principal actual
# ──────────────────────────────────────────
_CLAUDE_MODEL = "claude-sonnet-4-20250514"
_CLAUDE_API   = "https://api.anthropic.com/v1/messages"

_PROMPT = """Analiza esta imagen de comida y responde UNICAMENTE con JSON valido.
Sin texto adicional, sin backticks, sin explicaciones.

Estructura exacta requerida:
{
  "food_class": "nombre clave minusculas sin acentos (ej: tacos, pizza, arroz con pollo)",
  "display_name": "Nombre bonito en espanol con mayuscula inicial",
  "description": "Una linea describiendo el platillo e ingredientes principales",
  "confidence": 0.95,
  "calories": 320,
  "protein": 18.5,
  "carbs": 28.0,
  "fat": 12.0,
  "fiber": 3.5,
  "alternatives": [
    {
      "food_class": "segunda opcion clave",
      "display_name": "Segunda opcion en espanol",
      "confidence": 0.6,
      "calories": 280,
      "protein": 14.0,
      "carbs": 32.0,
      "fat": 9.0,
      "fiber": 2.0
    },
    {
      "food_class": "tercera opcion clave",
      "display_name": "Tercera opcion en espanol",
      "confidence": 0.3,
      "calories": 210,
      "protein": 10.0,
      "carbs": 25.0,
      "fat": 7.5,
      "fiber": 1.5
    }
  ]
}

REGLAS:
- Valores nutricionales son para {grams}g de porcion
- Estima con base en ingredientes visibles en la imagen
- Si no identificas la comida usa food_class: desconocido
- confidence va de 0.0 a 1.0
"""


def _image_to_base64(image_path: str) -> tuple[str, str]:
    ext = os.path.splitext(image_path)[1].lower()
    media_types = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }
    media_type = media_types.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def _predict_claude(image_path: str, grams: float = 150.0) -> list[dict]:
    """Llama a Claude Vision para identificar comida y obtener macros."""
    import urllib.request
    import urllib.error

    api_key = _get_api_key()
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY no configurada — usando fallback simulado")
        return []

    try:
        image_data, media_type = _image_to_base64(image_path)
    except Exception as e:
        logger.error("Error leyendo imagen: %s", e)
        return []

    prompt = _PROMPT.replace("{grams}", str(int(grams)))

    payload = json.dumps({
        "model": _CLAUDE_MODEL,
        "max_tokens": 1000,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type":       "base64",
                        "media_type": media_type,
                        "data":       image_data,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    }).encode("utf-8")

    req = urllib.request.Request(
        _CLAUDE_API,
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        logger.error("Claude API HTTP %s: %s", e.code, body)
        return []
    except Exception as e:
        logger.error("Error llamando Claude API: %s", e)
        return []

    raw_text = ""
    for block in response_data.get("content", []):
        if block.get("type") == "text":
            raw_text += block["text"]

    clean = raw_text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error("Error parseando JSON de Claude: %s\nRaw: %s", e, raw_text[:300])
        return []

    results = []

    # Resultado principal
    results.append({
        "food_class":     parsed.get("food_class", "desconocido"),
        "display_name":   parsed.get("display_name", "Desconocido"),
        "original_class": parsed.get("food_class", "desconocido"),
        "confidence":     float(parsed.get("confidence", 0.9)),
        "source":         "claude_vision",
        "description":    parsed.get("description", ""),
        "calories":       float(parsed.get("calories", 0)),
        "protein":        float(parsed.get("protein", 0)),
        "carbs":          float(parsed.get("carbs", 0)),
        "fat":            float(parsed.get("fat", 0)),
        "fiber":          float(parsed.get("fiber", 0)),
    })

    # Alternativas (máximo 2)
    for alt in parsed.get("alternatives", [])[:2]:
        results.append({
            "food_class":     alt.get("food_class", "desconocido"),
            "display_name":   alt.get("display_name", "Desconocido"),
            "original_class": alt.get("food_class", "desconocido"),
            "confidence":     float(alt.get("confidence", 0.3)),
            "source":         "claude_vision",
            "description":    "",
            "calories":       float(alt.get("calories", 0)),
            "protein":        float(alt.get("protein", 0)),
            "carbs":          float(alt.get("carbs", 0)),
            "fat":            float(alt.get("fat", 0)),
            "fiber":          float(alt.get("fiber", 0)),
        })

    logger.info(
        "Claude Vision: %s (%.0f%%) — %.0f kcal",
        results[0]["display_name"],
        results[0]["confidence"] * 100,
        results[0]["calories"],
    )
    return results


# ──────────────────────────────────────────
# 4. FALLBACK SIMULADO — solo desarrollo
# ──────────────────────────────────────────
def _predict_simulated() -> list[dict]:
    """Fallback aleatorio cuando no hay modelo ni API key."""
    from app.nutrition.food_table import get_food_classes, estimate_nutrition

    logger.warning("Prediccion SIMULADA — configura ANTHROPIC_API_KEY para resultados reales")

    classes = get_food_classes()
    top3    = random.sample(classes, min(3, len(classes)))
    confs   = sorted([random.uniform(0.3, 0.95) for _ in top3], reverse=True)

    results = []
    for cls, conf in zip(top3, confs):
        nutrition = estimate_nutrition(cls, "mediana") or {}
        results.append({
            "food_class":     cls,
            "display_name":   cls.title(),
            "original_class": cls,
            "confidence":     round(conf, 2),
            "source":         "simulado",
            "description":    "Modo simulado — configura tu API key",
            "calories":       nutrition.get("calories", 0),
            "protein":        nutrition.get("protein", 0),
            "carbs":          nutrition.get("carbs", 0),
            "fat":            nutrition.get("fat", 0),
            "fiber":          0.0,
        })
    return results


# ──────────────────────────────────────────
# 5. API PÚBLICA
# ──────────────────────────────────────────
def predict_food(image_path: str, grams: float = 150.0) -> list[dict]:
    """
    Punto de entrada principal.

    Orden de prioridad:
      1. Modelo PyTorch local (cuando esté entrenado)
      2. Claude Vision API   (motor actual)
      3. Fallback simulado   (desarrollo sin internet)

    Retorna lista de hasta 3 dicts con:
        food_class, display_name, original_class, confidence,
        source, description, calories, protein, carbs, fat, fiber
    """
    if _try_load_model():
        results = _predict_pytorch(image_path)
        if results:
            return results

    results = _predict_claude(image_path, grams=grams)
    if results:
        return results

    return _predict_simulated()


def is_model_loaded() -> bool:
    """True si el modelo PyTorch local está disponible."""
    _try_load_model()
    return _model_loaded


def get_inference_source() -> str:
    """Devuelve qué motor está activo: 'pytorch', 'claude_vision' o 'simulado'."""
    if _try_load_model():
        return "pytorch"
    if _get_api_key():
        return "claude_vision"
    return "simulado"