import hashlib
import os
import psycopg2
import psycopg2.extras
from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty, BooleanProperty, NumericProperty

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "127.0.0.1"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",     "base_principal"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "Lenovo2311"),
}


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _conn():
    c = psycopg2.connect(**DB_CONFIG)
    c.autocommit = False
    return c


def init_users_table():
    c = _conn()
    cur = c.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id        SERIAL PRIMARY KEY,
            nombre    VARCHAR(100) NOT NULL,
            apellido  VARCHAR(150),
            usuario   VARCHAR(255) NOT NULL UNIQUE,
            email     VARCHAR(150) NOT NULL UNIQUE,
            edad      INTEGER,
            password  VARCHAR(255) NOT NULL,
            fecha     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS perfil_fisico (
            usuario_id INTEGER PRIMARY KEY REFERENCES usuarios(id) ON DELETE CASCADE,
            altura     DOUBLE PRECISION,
            peso       DOUBLE PRECISION,
            imc        DOUBLE PRECISION
        )
    """)
    c.commit()
    cur.close()
    c.close()


class LoginScreen(Screen):
    error_msg = StringProperty("")

    def do_login(self, usuario: str, password: str):
        self.error_msg = ""
        usuario  = usuario.strip()
        password = password.strip()

        if not usuario or not password:
            self.error_msg = "Completa todos los campos"
            return

        try:
            c   = _conn()
            cur = c.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT * FROM usuarios WHERE usuario = %s AND password = %s",
                (usuario, _hash(password))
            )
            row = cur.fetchone()
            cur.close()
            c.close()
        except Exception as e:
            self.error_msg = f"Error de conexion: {e}"
            print(f"ERROR LOGIN: {e}")
            return

        if row:
            self.manager.current = "home"
        else:
            self.error_msg = "Usuario o contrasena incorrectos"

    def go_register(self):
        self.error_msg = ""
        self.manager.current = "register"


class RegisterScreen(Screen):
    msg        = StringProperty("")
    msg_ok     = BooleanProperty(False)
    usuario_id = NumericProperty(0)

    def do_register(self, nombre, apellido, usuario, email, edad, password, confirm=""):
        self.msg = ""
        print(f"DEBUG do_register llamado: {nombre}, {usuario}, {email}")

        nombre   = nombre.strip()
        apellido = apellido.strip()
        usuario  = usuario.strip()
        email    = email.strip()
        edad     = edad.strip()

        if not nombre or not usuario or not email or not password:
            self.msg    = "Todos los campos son obligatorios"
            self.msg_ok = False
            return

        if len(password) < 6:
            self.msg    = "La contrasena debe tener al menos 6 caracteres"
            self.msg_ok = False
            return

        if confirm and password != confirm:
            self.msg    = "Las contrasenas no coinciden"
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
            print("DEBUG: intentando conectar a PostgreSQL...")
            c   = _conn()
            cur = c.cursor()
            cur.execute(
                """INSERT INTO usuarios (nombre, apellido, usuario, email, edad, password)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (nombre, apellido, usuario, email, edad_int, _hash(password))
            )
            self.usuario_id = cur.fetchone()[0]
            c.commit()
            cur.close()
            c.close()
            print(f"DEBUG: usuario creado con id={self.usuario_id}")

            self.msg    = "Cuenta creada! Ingresa tus datos corporales"
            self.msg_ok = True

            from kivy.clock import Clock
            Clock.schedule_once(lambda dt: self._go_setup(), 1.0)

        except psycopg2.errors.UniqueViolation:
            self.msg    = "El usuario o email ya existe"
            self.msg_ok = False
            print("DEBUG: UniqueViolation - usuario o email duplicado")
        except Exception as e:
            self.msg    = f"Error: {e}"
            self.msg_ok = False
            print(f"ERROR REGISTRO: {e}")

    def _go_setup(self):
        setup = self.manager.get_screen("profile_setup")
        setup.usuario_id = self.usuario_id
        self.manager.current = "profile_setup"

    def go_back(self):
        self.msg = ""
        self.manager.current = "login"


class ProfileSetupScreen(Screen):
    msg        = StringProperty("")
    msg_ok     = BooleanProperty(False)
    usuario_id = NumericProperty(0)

    def do_save(self, altura: str, peso: str):
        self.msg = ""
        altura = altura.strip()
        peso   = peso.strip()

        if not altura or not peso:
            self.msg    = "Completa altura y peso"
            self.msg_ok = False
            return

        try:
            altura_f = float(altura.replace(",", "."))
            peso_f   = float(peso.replace(",", "."))
        except ValueError:
            self.msg    = "Ingresa numeros validos"
            self.msg_ok = False
            return

        if not (1.0 <= altura_f <= 2.5):
            self.msg    = "Altura debe estar entre 1.0 y 2.5 metros"
            self.msg_ok = False
            return

        if not (20 <= peso_f <= 300):
            self.msg    = "Peso debe estar entre 20 y 300 kg"
            self.msg_ok = False
            return

        imc = round(peso_f / (altura_f ** 2), 2)

        try:
            c   = _conn()
            cur = c.cursor()
            cur.execute("""
                INSERT INTO perfil_fisico (usuario_id, altura, peso, imc)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (usuario_id) DO UPDATE
                    SET altura = EXCLUDED.altura,
                        peso   = EXCLUDED.peso,
                        imc    = EXCLUDED.imc
            """, (self.usuario_id, altura_f, peso_f, imc))
            c.commit()
            cur.close()
            c.close()

            self.msg    = f"Perfil guardado! Tu IMC es {imc}"
            self.msg_ok = True

            from kivy.clock import Clock
            Clock.schedule_once(lambda dt: self._go_home(), 1.5)

        except Exception as e:
            self.msg    = f"Error al guardar: {e}"
            self.msg_ok = False
            print(f"ERROR PERFIL: {e}")

    def _go_home(self):
        self.manager.current = "home"

    def go_skip(self):
        self.manager.current = "home"
