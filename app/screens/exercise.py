"""
Pantalla de ejercicio con cronometro y calculo de calorias.
Toma el peso del perfil de usuario.
Modos: libre (cuenta hacia arriba) y meta (cuenta hacia una meta de tiempo).
"""

from kivy.uix.screenmanager import Screen
from kivy.properties import (NumericProperty, StringProperty,
                              BooleanProperty, OptionProperty)
from kivy.clock import Clock


# MET (Metabolic Equivalent) por ejercicio
# Calorias/min = MET x peso_kg / 60 x 3.5 / 200  (formula simplificada)
MET = {
    "caminar": 3.5,
    "correr":  9.0,
    "saltar":  8.0,
}


class ExerciseScreen(Screen):

    # Ejercicio seleccionado desde HomeScreen
    selected_exercise = OptionProperty("correr", options=["caminar", "correr", "saltar"])

    # Cronometro
    elapsed_secs  = NumericProperty(0)
    timer_text    = StringProperty("00:00:00")
    is_running    = BooleanProperty(False)

    # Modo: "libre" sube, "meta" cuenta hacia meta
    mode          = OptionProperty("libre", options=["libre", "meta"])
    goal_secs     = NumericProperty(1800)   # meta default: 30 min
    goal_text     = StringProperty("30:00")
    remaining_text = StringProperty("30:00")

    # Calorias y distancia
    calories_burned = NumericProperty(0.0)
    distance_km     = NumericProperty(0.0)

    # Velocidades promedio km/h por ejercicio
    _SPEEDS = {"caminar": 5.0, "correr": 9.0, "saltar": 2.0}

    _clock_event = None

    def on_pre_enter(self):
        self.reset_timer()

    def on_leave(self):
        self._stop_clock()
        # Actualizar HomeScreen con calorias quemadas y distancia
        try:
            home = self.manager.get_screen("home")
            home.burned_calories = round(self.calories_burned, 1)
            home.distance_today  = round(self.distance_km, 2)
        except Exception:
            pass

    # ── Peso del usuario desde perfil ──
    def _get_weight(self) -> float:
        try:
            profile = self.manager.get_screen("profile")
            return float(profile.weight) if profile.weight else 70.0
        except Exception:
            return 70.0

    # ── Control del cronometro ──
    def toggle_timer(self):
        if self.is_running:
            self._stop_clock()
        else:
            self._start_clock()

    def _start_clock(self):
        self.is_running = True
        self._clock_event = Clock.schedule_interval(self._tick, 1)

    def _stop_clock(self):
        self.is_running = False
        if self._clock_event:
            self._clock_event.cancel()
            self._clock_event = None

    def _tick(self, dt):
        self.elapsed_secs += 1
        self._update_displays()

        # Si modo meta y llego a la meta, parar
        if self.mode == "meta" and self.elapsed_secs >= self.goal_secs:
            self._stop_clock()

    def _update_displays(self):
        # Texto cronometro
        self.timer_text = self._fmt(self.elapsed_secs)

        # Restante en modo meta
        remaining = max(0, self.goal_secs - self.elapsed_secs)
        self.remaining_text = self._fmt(remaining)

        # Calorias: MET x peso x horas
        weight  = self._get_weight()
        met     = MET.get(self.selected_exercise, 5.0)
        hours   = self.elapsed_secs / 3600.0
        self.calories_burned = met * weight * hours

        # Distancia
        speed = self._SPEEDS.get(self.selected_exercise, 5.0)
        self.distance_km = speed * hours

    def reset_timer(self):
        self._stop_clock()
        self.elapsed_secs    = 0
        self.calories_burned = 0.0
        self.distance_km     = 0.0
        self.timer_text      = "00:00:00"
        self._update_goal_text()
        self.remaining_text  = self.goal_text

    # ── Modo ──
    def set_mode(self, mode: str):
        self.mode = mode
        self.reset_timer()

    # ── Meta de tiempo ──
    def set_goal(self, minutes: int):
        self.goal_secs = minutes * 60
        self._update_goal_text()
        self.remaining_text = self.goal_text
        if not self.is_running:
            self.elapsed_secs = 0
            self.calories_burned = 0.0

    def _update_goal_text(self):
        m = self.goal_secs // 60
        s = self.goal_secs % 60
        self.goal_text = f"{m:02d}:{s:02d}"

    # ── Utilidad ──
    @staticmethod
    def _fmt(secs: int) -> str:
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ── Navegacion ──
    def go_back(self):
        self.manager.current = "home"
