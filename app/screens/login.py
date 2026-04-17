import sqlite3
import hashlib
import os
from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty, BooleanProperty


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "calorias.db")


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_users_table():
    c = _conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre    VARCHAR(100) NOT NULL,
            apellido  VARCHAR(150),
            usuario   VARCHAR(255) NOT NULL UNIQUE,
            email     VARCHAR(150) NOT NULL UNIQUE,
            edad      INTEGER,
            password  VARCHAR(255) NOT NULL,
            Fecha     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.commit()
    c.close()


class LoginScreen(Screen):
    error_msg = StringProperty("")

    def do_login(self, usuario: str, password: str):
        self.error_msg = ""
        usuario = usuario.strip()
        password = password.strip()

        if not usuario or not password:
            self.error_msg = "Completa todos los campos"
            return

        c = _conn()
        row = c.execute(
            "SELECT * FROM usuarios WHERE usuario = ? AND password = ?",
            (usuario, _hash(password))
        ).fetchone()
        c.close()

        if row:
            self.manager.current = "home"
        else:
            self.error_msg = "Usuario o contrasena incorrectos"

    def go_register(self):
        self.error_msg = ""
        self.manager.current = "register"


class RegisterScreen(Screen):
    msg    = StringProperty("")
    msg_ok = BooleanProperty(False)

    def do_register(self, nombre, apellido, usuario, email, edad):
        self.msg = ""
        nombre  = nombre.strip()
        apellido = apellido.strip()
        usuario = usuario.strip()
        email   = email.strip()
        edad    = edad.strip()

        if not nombre or not usuario or not email:
            self.msg    = "Nombre, usuario y email son obligatorios"
            self.msg_ok = False
            return

        if len(usuario) < 3:
            self.msg    = "El usuario debe tener al menos 3 caracteres"
            self.msg_ok = False
            return

        if "@" not in email:
            self.msg    = "Correo electronico invalido"
            self.msg_ok = False
            return

        edad_int = int(edad) if edad.isdigit() else None

        try:
            c = _conn()
            c.execute(
                """INSERT INTO usuarios (nombre, apellido, usuario, email, edad, password)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (nombre, apellido, usuario, email, edad_int, _hash("1234"))
            )
            c.commit()
            c.close()
            self.msg    = "Cuenta creada! Ya puedes iniciar sesion"
            self.msg_ok = True
        except sqlite3.IntegrityError:
            self.msg    = "El usuario o email ya existe"
            self.msg_ok = False

    def go_back(self):
        self.msg = ""
        self.manager.current = "login"
