from kivy.uix.screenmanager import Screen
from kivy.properties import (StringProperty, NumericProperty, ListProperty,
                              BooleanProperty)
from app.services.inference import predict_food, is_model_loaded
from app.nutrition.food_table import estimate_nutrition, get_food_classes
from app.database.db import save_meal


class ResultScreen(Screen):
    image_path = StringProperty("")

    # Predicción
    predictions = ListProperty([])
    selected_class = StringProperty("")
    display_name = StringProperty("")
    confidence_text = StringProperty("")
    model_status = StringProperty("")

    # Porción
    portion = StringProperty("mediana")
    custom_grams = StringProperty("")

    # Resultado nutricional
    calories = NumericProperty(0)
    protein = NumericProperty(0)
    carbs = NumericProperty(0)
    fat = NumericProperty(0)
    grams_display = NumericProperty(0)

    saved = BooleanProperty(False)

    def on_pre_enter(self):
        self.saved = False
        self._run_prediction()

    def _run_prediction(self):
        self.model_status = "IA" if is_model_loaded() else "Simulado"
        results = predict_food(self.image_path)
        self.predictions = results
        if results:
            self._select(results[0])
        self._recalculate()

    def _select(self, pred):
        self.selected_class = pred["food_class"]
        self.display_name = pred.get("display_name", pred["food_class"].title())
        conf = pred["confidence"]
        self.confidence_text = f"Confianza: {int(conf * 100)}%"

    def select_prediction(self, index):
        if 0 <= index < len(self.predictions):
            self._select(self.predictions[index])
            self._recalculate()

    def set_portion(self, portion):
        self.portion = portion
        self.custom_grams = ""
        self._recalculate()

    def apply_custom_grams(self):
        self._recalculate()

    def _recalculate(self):
        grams = None
        if self.custom_grams.strip():
            try:
                grams = float(self.custom_grams.strip())
            except ValueError:
                pass

        result = estimate_nutrition(self.selected_class, self.portion, grams)
        if result:
            self.calories = result["calories"]
            self.protein = result["protein"]
            self.carbs = result["carbs"]
            self.fat = result["fat"]
            self.grams_display = result["grams"]

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

    def go_home(self):
        self.manager.current = "home"

    def go_back(self):
        self.manager.current = "capture"
