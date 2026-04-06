from kivy.uix.screenmanager import Screen
from kivy.properties import NumericProperty, StringProperty, OptionProperty
from app.database.db import get_meals_today


class HomeScreen(Screen):
    # Nutrición
    total_calories  = NumericProperty(0)
    total_protein   = NumericProperty(0)
    total_carbs     = NumericProperty(0)
    total_fat       = NumericProperty(0)
    meals_count     = NumericProperty(0)

    # Ejercicio quemado (lo actualizará ExerciseScreen al terminar sesión)
    burned_calories = NumericProperty(0)

    # Ejercicio seleccionado en home: "caminar" | "correr" | "saltar"
    selected_exercise = OptionProperty("correr", options=["caminar", "correr", "saltar"])

    # Distancia acumulada hoy (km) — alimentada desde ExerciseScreen
    distance_today  = NumericProperty(0.0)

    def on_pre_enter(self):
        self.refresh_summary()

    def refresh_summary(self):
        meals = get_meals_today()
        self.meals_count     = len(meals)
        self.total_calories  = round(sum(m["calories"] for m in meals), 1)
        self.total_protein   = round(sum(m["protein"]  for m in meals), 1)
        self.total_carbs     = round(sum(m["carbs"]    for m in meals), 1)
        self.total_fat       = round(sum(m["fat"]      for m in meals), 1)

    def select_exercise(self, exercise: str):
        self.selected_exercise = exercise

    def go_capture(self):
        self.manager.current = "capture"

    def go_history(self):
        self.manager.current = "history"

    def go_exercise(self):
        """Pasa el ejercicio seleccionado a la pantalla de ejercicio y navega."""
        ex_screen = self.manager.get_screen("exercise")
        ex_screen.selected_exercise = self.selected_exercise
        self.manager.current = "exercise"

    def go_profile(self):
        self.manager.current = "profile"
