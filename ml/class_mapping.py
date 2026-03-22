"""
Mapeo de clases Food-101 (inglés) a nombres en español
y a las claves de nuestra tabla nutricional.

Food-101 tiene 101 clases. Mapeamos las más relevantes a nuestro
food_table y el resto se mapea con nombre genérico para mostrar al usuario.
"""

# Mapeo: clase_food101 -> (nombre_español, clave_food_table o None)
# Si clave_food_table es None, usaremos valores nutricionales genéricos.

CLASS_MAP: dict[str, tuple[str, str | None]] = {
    # ─── Exactas o casi exactas con nuestra tabla ───
    "pizza":                    ("Pizza",               "pizza"),
    "hamburger":                ("Hamburguesa",         "hamburguesa"),
    "hot_dog":                  ("Hot Dog",             "hot dog"),
    "french_fries":             ("Papas fritas",        "papas fritas"),
    "chicken_wings":            ("Alitas de pollo",     "nuggets"),
    "tacos":                    ("Tacos",               "tacos"),
    "burritos":                 ("Burrito",             "burrito"),
    "enchiladas":               ("Enchiladas",          "enchiladas"),
    "nachos":                   ("Nachos",              "tostadas"),
    "fried_rice":               ("Arroz frito",         "arroz blanco"),
    "grilled_salmon":           ("Salmón a la plancha", "pescado a la plancha"),
    "fish_and_chips":           ("Pescado frito",       "pescado frito"),
    "spaghetti_bolognese":      ("Spaghetti",           "spaghetti"),
    "spaghetti_carbonara":      ("Spaghetti carbonara", "spaghetti"),
    "lasagna":                  ("Lasagna",             "lasagna"),
    "sushi":                    ("Sushi",               "sushi"),
    "ramen":                    ("Ramen",               "ramen"),
    "caesar_salad":             ("Ensalada César",      "ensalada cesar"),
    "greek_salad":              ("Ensalada griega",     "ensalada"),
    "eggs_benedict":            ("Huevos benedictinos", "huevos fritos"),
    "omelette":                 ("Omelette",            "huevos revueltos"),
    "pancakes":                 ("Pancakes",            "pancakes"),
    "french_toast":             ("Pan francés",         "pan tostado"),
    "donuts":                   ("Dona",                "dona"),
    "churros":                  ("Churros",             "churros"),
    "ice_cream":                ("Helado",              "helado"),
    "chocolate_cake":           ("Pastel de chocolate", "pastel"),
    "cheesecake":               ("Cheesecake",          "pastel"),
    "strawberry_shortcake":     ("Pastel de fresa",     "pastel"),
    "apple_pie":                ("Pay de manzana",      "pastel"),
    "carrot_cake":              ("Pastel de zanahoria", "pastel"),
    "cup_cakes":                ("Cupcakes",            "pastel"),
    "macarons":                 ("Macarons",            "galletas"),
    "club_sandwich":            ("Club sándwich",       "sandwich"),
    "grilled_cheese_sandwich":  ("Sándwich de queso",   "sandwich"),
    "pulled_pork_sandwich":     ("Sándwich de cerdo",   "sandwich"),
    "breakfast_burrito":        ("Burrito de desayuno",  "burrito"),
    "chicken_quesadilla":       ("Quesadilla de pollo", "quesadilla"),
    "steak":                    ("Carne asada",         "carne asada"),
    "filet_mignon":             ("Filete mignon",       "carne asada"),
    "pork_chop":                ("Chuleta de cerdo",    "carne asada"),
    "prime_rib":                ("Costilla prime",      "carne asada"),
    "fried_chicken":            ("Pollo frito",         "pollo asado"),
    "chicken_curry":            ("Pollo al curry",      "arroz con pollo"),
    "bibimbap":                 ("Bibimbap",            "arroz con pollo"),
    "pad_thai":                 ("Pad thai",            "spaghetti"),
    "pho":                      ("Pho",                 "ramen"),
    "french_onion_soup":        ("Sopa de cebolla",     "sopa de verduras"),
    "clam_chowder":             ("Sopa de almeja",      "sopa de verduras"),
    "miso_soup":                ("Sopa miso",           "sopa de verduras"),
    "hot_and_sour_soup":        ("Sopa agripicante",    "sopa de verduras"),
    "waffles":                  ("Waffles",             "pancakes"),
    "bread_pudding":            ("Budín de pan",        "pastel"),
    "creme_brulee":             ("Crème brûlée",        "helado"),
    "tiramisu":                 ("Tiramisú",            "pastel"),
    "chocolate_mousse":         ("Mousse de chocolate", "helado"),
    "frozen_yogurt":            ("Yogurt helado",       "helado"),
    "baklava":                  ("Baklava",             "churros"),
    "cannoli":                  ("Cannoli",             "churros"),
    "bruschetta":               ("Bruschetta",          "pan tostado"),
    "spring_rolls":             ("Rollos primavera",    "nuggets"),
    "samosa":                   ("Samosa",              "nuggets"),
    "falafel":                  ("Falafel",             "nuggets"),
    "onion_rings":              ("Aros de cebolla",     "papas fritas"),
    "mozzarella_sticks":        ("Palitos de queso",    "nuggets"),
    "garlic_bread":             ("Pan de ajo",          "pan tostado"),
    "ceviche":                  ("Ceviche",             "pescado a la plancha"),
    "gyoza":                    ("Gyoza",               "nuggets"),
    "dumplings":                ("Dumplings",           "nuggets"),
    "edamame":                  ("Edamame",             "ensalada"),
    "hummus":                   ("Hummus",              "ensalada"),
    "guacamole":                ("Guacamole",           "ensalada"),

    # ─── Clases sin mapeo nutricional directo ───
    "baby_back_ribs":           ("Costillas BBQ",       "carne asada"),
    "beef_carpaccio":           ("Carpaccio",           "carne asada"),
    "beef_tartare":             ("Tártara de res",      "carne asada"),
    "beet_salad":               ("Ensalada de betabel", "ensalada"),
    "beignets":                 ("Beignets",            "dona"),
    "caprese_salad":            ("Ensalada caprese",    "ensalada"),
    "cheese_plate":             ("Tabla de quesos",     "quesadilla"),
    "chicken_curry":            ("Curry de pollo",      "arroz con pollo"),
    "crab_cakes":               ("Pasteles de cangrejo","pescado frito"),
    "croque_madame":            ("Croque madame",       "sandwich"),
    "deviled_eggs":             ("Huevos rellenos",     "huevos revueltos"),
    "escargots":                ("Escargots",           "pescado a la plancha"),
    "foie_gras":                ("Foie gras",           "carne asada"),
    "lobster_bisque":           ("Bisque de langosta",  "sopa de verduras"),
    "lobster_roll_sandwich":    ("Sándwich de langosta","sandwich"),
    "macaroni_and_cheese":      ("Mac and cheese",      "spaghetti"),
    "mussels":                  ("Mejillones",          "pescado a la plancha"),
    "oysters":                  ("Ostiones",            "pescado a la plancha"),
    "paella":                   ("Paella",              "arroz con pollo"),
    "peking_duck":              ("Pato pekín",          "pollo asado"),
    "panna_cotta":              ("Panna cotta",         "helado"),
    "poutine":                  ("Poutine",             "papas fritas"),
    "ravioli":                  ("Ravioli",             "spaghetti"),
    "red_velvet_cake":          ("Red velvet",          "pastel"),
    "risotto":                  ("Risotto",             "arroz blanco"),
    "scallops":                 ("Callos de hacha",     "pescado a la plancha"),
    "seaweed_salad":            ("Ensalada de algas",   "ensalada"),
    "shrimp_and_grits":         ("Camarones con grits", "pescado frito"),
    "sashimi":                  ("Sashimi",             "sushi"),
    "takoyaki":                 ("Takoyaki",            "nuggets"),
    "tuna_tartare":             ("Tártara de atún",     "sushi"),
}


def food101_to_spanish(class_name: str) -> str:
    """Convierte nombre Food-101 a español."""
    if class_name in CLASS_MAP:
        return CLASS_MAP[class_name][0]
    return class_name.replace("_", " ").title()


def food101_to_food_table_key(class_name: str) -> str | None:
    """Convierte nombre Food-101 a clave de nuestra food_table."""
    if class_name in CLASS_MAP:
        return CLASS_MAP[class_name][1]
    return None


def get_mapped_classes() -> list[str]:
    """Devuelve las clases de Food-101 que tenemos mapeadas."""
    return sorted(CLASS_MAP.keys())
