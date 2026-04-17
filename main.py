"""
Kaguar-ia — Punto de entrada principal.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, SlideTransition
from kivy.core.window import Window

from app.screens.home import HomeScreen
from app.screens.capture import CaptureScreen
from app.screens.result import ResultScreen
from app.screens.history import HistoryScreen
from app.screens.exercise import ExerciseScreen
from app.screens.login import LoginScreen, RegisterScreen, init_users_table
from app.database.db import init_db

Window.size = (390, 750)

KV_DIR = os.path.join(ROOT_DIR, "kv")
for kv_file in ["login.kv", "home.kv", "capture.kv", "result.kv", "history.kv", "exercise.kv"]:
    Builder.load_file(os.path.join(KV_DIR, kv_file))


class CaloriasApp(App):
    title = "Kaguar-ia"

    def build(self):
        init_db()
        init_users_table()

        sm = ScreenManager(transition=SlideTransition())
        sm.add_widget(LoginScreen(name="login"))
        sm.add_widget(RegisterScreen(name="register"))
        sm.add_widget(HomeScreen(name="home"))
        sm.add_widget(CaptureScreen(name="capture"))
        sm.add_widget(ResultScreen(name="result"))
        sm.add_widget(HistoryScreen(name="history"))
        sm.add_widget(ExerciseScreen(name="exercise"))

        return sm


if __name__ == "__main__":
    CaloriasApp().run()
