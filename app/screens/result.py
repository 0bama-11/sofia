from kivy.uix.screenmanager import Screen
from kivy.properties import (StringProperty, NumericProperty, ListProperty,
                              BooleanProperty)
from app.services.inference import predict_food, is_model_loaded, get_inference_source
from app.nutrition.food_table import estimate_nutrition
from app.database.db import save_meal


class ResultScreen(Screen):
    image_path = StringProperty("")

    # Predicción
    predictions    = ListProperty([])
    selected_index = NumericProperty(0)
    selected_class = StringProperty("")
    display_name   = StringProperty("")
    description    = StringProperty("")
    confidence_text = StringProperty("")
    model_status   = StringProperty("")

    # Porción
    portion      = StringProperty("mediana")
    custom_grams = StringProperty("")

    # Resultado nutricional
    calories      = NumericProperty(0)
    protein       = NumericProperty(0)
    carbs         = NumericProperty(0)
    fat           = NumericProperty(0)
    fiber         = NumericProperty(0)
    grams_display = NumericProperty(0)

    saved = BooleanProperty(False)

    # ── Constantes de porción base (gramos) ──
    _PORTION_GRAMS = {"pequeña": 112.5, "mediana": 150.0, "grande": 225.0}

    def on_pre_enter(self):
        self.saved = False
        self._run_prediction()

    # ─────────────────────────────────────────
    # Predicción
    # ─────────────────────────────────────────
    def _run_prediction(self):
        source = get_inference_source()
        labels = {
            "pytorch":      "Modelo local",
            "claude_vision": "Claude Vision IA",
            "simulado":     "Modo simulado",
        }
        self.model_status = labels.get(source, "IA")

        grams = self._current_grams()
        results = predict_food(self.image_path, grams=grams)
        self.predictions = results

        if results:
            self.selected_index = 0
            self._apply_prediction(results[0])

    def _apply_prediction(self, pred: dict):
        """Aplica un resultado de predicción a las propiedades de pantalla."""
        self.selected_class  = pred["food_class"]
        self.display_name    = pred.get("display_name", pred["food_class"].title())
        self.description     = pred.get("description", "")
        conf                 = pred["confidence"]
        self.confidence_text = f"Confianza: {int(conf * 100)}%"
        self._recalculate(pred)

    def select_prediction(self, index: int):
        """Llamado desde la UI cuando el usuario elige una alternativa."""
        if 0 <= index < len(self.predictions):
            self.selected_index = index
            self._apply_prediction(self.predictions[index])

    # ─────────────────────────────────────────
    # Porción
    # ─────────────────────────────────────────
    def set_portion(self, portion: str):
        self.portion = portion
        self.custom_grams = ""
        pred = self._selected_pred()
        if pred:
            self._recalculate(pred)

    def apply_custom_grams(self):
        pred = self._selected_pred()
        if pred:
            self._recalculate(pred)

    def _current_grams(self) -> float:
        if self.custom_grams.strip():
            try:
                return float(self.custom_grams.strip())
            except ValueError:
                pass
        return self._PORTION_GRAMS.get(self.portion, 150.0)

    # ─────────────────────────────────────────
    # Recálculo nutricional
    # ─────────────────────────────────────────
    def _recalculate(self, pred: dict):
        """
        Si el resultado viene de Claude Vision, escala los macros al
        peso real de la porción (Claude ya devuelve valores por porción base).
        Si viene de PyTorch (sin macros), usa food_table como antes.
        """
        grams = self._current_grams()

        source = pred.get("source", "simulado")

        if source == "claude_vision" and pred.get("calories") is not None:
            # Claude ya calculó para la porción original (150g base).
            # Escalamos si el usuario cambió el peso.
            base_grams = 150.0
            factor = grams / base_grams
            self.calories      = round(pred["calories"]  * factor, 1)
            self.protein       = round(pred["protein"]   * factor, 1)
            self.carbs         = round(pred["carbs"]     * factor, 1)
            self.fat           = round(pred["fat"]       * factor, 1)
            self.fiber         = round(pred.get("fiber", 0) * factor, 1)
            self.grams_display = grams

        else:
            # PyTorch o simulado → food_table.py como antes
            result = estimate_nutrition(self.selected_class, self.portion,
                                        grams if self.custom_grams.strip() else None)
            if result:
                self.calories      = result["calories"]
                self.protein       = result["protein"]
                self.carbs         = result["carbs"]
                self.fat           = result["fat"]
                self.fiber         = 0.0
                self.grams_display = result["grams"]

    # ─────────────────────────────────────────
    # Guardar
    # ─────────────────────────────────────────
    def save_result(self):
        if self.saved:
            return
        save_meal(
            food_class=self.selected_class,
            portion=self.portion,
            grams=self.grams_display,
            calories=self.calories,
            protein=self.protein,
            carbs=self.carbs,
            fat=self.fat,
            image_path=self.image_path,
        )
        self.saved = True

    # ─────────────────────────────────────────
    # Navegación
    # ─────────────────────────────────────────
    def go_home(self):
        self.manager.current = "home"

    def go_back(self):
        self.manager.current = "capture"

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────
    def _selected_pred(self) -> dict | None:
        if self.predictions and 0 <= self.selected_index < len(self.predictions):
            return self.predictions[self.selected_index]
        return None