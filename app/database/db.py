import psycopg2
import psycopg2.extras
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURACION DE CONEXION A POSTGRESQL
# Cambia estos datos si tu servidor cambia
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     5432,
    "user":     "postgres",
    "password": "Lenovo2311",
    "dbname":   "postgres"
}


def _get_connection():
    """
    Crea y retorna una conexion a PostgreSQL.
    RealDictCursor hace que cada fila sea un diccionario,
    igual que sqlite3.Row que teniamos antes.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    return conn


def init_db():
    """
    Crea la tabla meals si no existe.
    En PostgreSQL usamos SERIAL en lugar de INTEGER AUTOINCREMENT,
    y los tipos de datos son ligeramente diferentes.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meals (
            id          SERIAL PRIMARY KEY,
            food_class  TEXT NOT NULL,
            portion     TEXT NOT NULL DEFAULT 'mediana',
            grams       REAL NOT NULL DEFAULT 100,
            calories    REAL NOT NULL,
            protein     REAL NOT NULL DEFAULT 0,
            carbs       REAL NOT NULL DEFAULT 0,
            fat         REAL NOT NULL DEFAULT 0,
            image_path  TEXT,
            created_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()


def save_meal(food_class, portion, grams, calories, protein, carbs, fat,
              image_path=None):
    """
    Guarda una comida en la base de datos.
    NOTA: En PostgreSQL los placeholders son %s en lugar de ?
    RETURNING id nos da el id del registro recien insertado.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO meals (food_class, portion, grams, calories, protein,
                           carbs, fat, image_path, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (food_class, portion, grams, calories, protein, carbs, fat,
          image_path, datetime.now().isoformat()))
    conn.commit()
    meal_id = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return meal_id


def get_meals_today():
    """
    Retorna todas las comidas del dia de hoy.
    Usamos RealDictCursor para que cada fila sea un dict,
    igual que antes con sqlite3.Row.
    """
    conn = _get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute(
        "SELECT * FROM meals WHERE created_at LIKE %s ORDER BY created_at DESC",
        (f"{today}%",)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()
    conn.close()
    return rows


def get_meals_history(limit=50):
    """
    Retorna el historial de comidas mas recientes.
    """
    conn = _get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cursor.execute(
        "SELECT * FROM meals ORDER BY created_at DESC LIMIT %s",
        (limit,)
    )
    rows = [dict(r) for r in cursor.fetchall()]
    cursor.close()
    conn.close()
    return rows


def delete_meal(meal_id):
    """
    Elimina una comida por su id.
    """
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM meals WHERE id = %s", (meal_id,))
    conn.commit()
    cursor.close()
    conn.close()
