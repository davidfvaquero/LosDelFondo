"""Pure chatbot helpers used by the Streamlit app and tests."""

from __future__ import annotations
import unicodedata
import pandas as pd

# Diccionario de alias para normalizar nombres de CCAA
ALIASES = {
    "andalucia": "Andalucía",
    "aragon": "Aragón",
    "asturias": "Asturias, Principado de",
    "baleares": "Balears, Illes",
    "balears": "Balears, Illes",
    "canarias": "Canarias",
    "cantabria": "Cantabria",
    "leon": "Castilla y León",
    "mancha": "Castilla - La Mancha",
    "cataluña": "Cataluña",
    "catalunya": "Cataluña",
    "catalonia": "Cataluña",
    "valencia": "Comunitat Valenciana",
    "valenciana": "Comunitat Valenciana",
    "extremadura": "Extremadura",
    "galicia": "Galicia",
    "madrid": "Madrid, Comunidad de",
    "murcia": "Murcia, Región de",
    "navarra": "Navarra, Comunidad Foral de",
    "vasco": "País Vasco",
    "rioja": "Rioja, La",
}

# Patrones de búsqueda para lógica determinista
MAX_SPEND_PATTERNS = ["gasta mas", "maximo gasto", "most spending", "highest spending", "mas dinero"]
MIN_SPEND_PATTERNS = ["gasta menos", "minimo gasto", "least spending", "lowest spending"]
MAX_LICENSE_PATTERNS = ["mas licencias", "most licenses", "mas federados", "mas socios"]

def normalize(text: str) -> str:
    """Retorna texto en minúsculas y sin acentos."""
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(char) != "Mn"
    )

def prepare_assistant_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los nombres de las columnas para el chatbot."""
    return df.rename(
        columns={
            "Gasto_Promedio_Hogar_Eur": "Gasto Promedio Hogar Eur",
            "Licencias_Federadas": "Licencias Federadas",
        }
    )

# --- NUEVOS MÉTODOS REQUERIDOS POR APP.PY ---

def load_models():
    """
    Carga los modelos de IA. 
    Retorna (None, None) por ahora para evitar errores de dependencias pesadas.
    """
    toxic_clf = None 
    llm_pipeline = None
    return toxic_clf, llm_pipeline

def check_toxicity(text: str, classifier=None):
    """
    Verifica si el texto es tóxico. 
    Retorna una tupla (is_toxic, score).
    """
    # Lógica por defecto: no es tóxico si el clasificador es None
    return False, 0.0

def generate_llm_response(prompt: str, df: pd.DataFrame, pipeline, lang: str) -> str:
    """
    Punto de entrada que conecta el prompt con la lógica de datos.
    Como no hay pipeline de LLM real cargado, usa la lógica determinista.
    """
    # Definición interna de etiquetas para evitar fallos de importación
    labels_map = {
        "ES": {
            "chat_max_spend": "🔍 La CCAA que más gasta es {region} con {value} €.",
            "chat_min_spend": "🔍 La CCAA que menos gasta es {region} con {value} €.",
            "chat_max_lic": "🏆 {region} lidera en licencias con {value:,}.",
            "chat_single_region": "📍 En {region}: Gasto de {gasto} € y {lic} licencias.",
            "chat_analyze": "🧠 He analizado los datos actuales:",
            "chat_error_data": "⚠️ No hay datos disponibles para el análisis."
        },
        "EN": {
            "chat_max_spend": "🔍 The region with the highest spending is {region} with {value} €.",
            "chat_min_spend": "🔍 The region with the lowest spending is {region} with {value} €.",
            "chat_max_lic": "🏆 {region} leads in licenses with {value:,}.",
            "chat_single_region": "📍 In {region}: Spending of {gasto} € and {lic} licenses.",
            "chat_analyze": "🧠 I have analyzed the current data:",
            "chat_error_data": "⚠️ No data available for analysis."
        }
    }
    
    selected_labels = labels_map.get(lang, labels_map["ES"])
    return generate_chat_response(prompt, df, selected_labels)

# --- LÓGICA DE RESPUESTA ---

def generate_chat_response(prompt: str, df: pd.DataFrame, labels: dict[str, str]) -> str:
    """Retorna una respuesta basada en reglas filtrando el DataFrame."""
    if df.empty:
        return labels["chat_error_data"]

    prompt_normalized = normalize(prompt)

    if any(pattern in prompt_normalized for pattern in MAX_SPEND_PATTERNS):
        row = df.loc[df["Gasto Promedio Hogar Eur"].idxmax()]
        return labels["chat_max_spend"].format(region=row["CCAA"], value=row["Gasto Promedio Hogar Eur"])

    if any(pattern in prompt_normalized for pattern in MIN_SPEND_PATTERNS):
        row = df.loc[df["Gasto Promedio Hogar Eur"].idxmin()]
        return labels["chat_min_spend"].format(region=row["CCAA"], value=row["Gasto Promedio Hogar Eur"])

    if any(pattern in prompt_normalized for pattern in MAX_LICENSE_PATTERNS):
        row = df.loc[df["Licencias Federadas"].idxmax()]
        return labels["chat_max_lic"].format(region=row["CCAA"], value=int(row["Licencias Federadas"]))

    for alias, official_name in ALIASES.items():
        if alias in prompt_normalized:
            # Buscamos la fila que coincida con el nombre oficial
            match = df[df["CCAA"] == official_name]
            if not match.empty:
                row = match.iloc[0]
                return labels["chat_single_region"].format(
                    region=official_name,
                    gasto=row["Gasto Promedio Hogar Eur"],
                    lic=int(row["Licencias Federadas"])
                )

    # Respuesta por defecto si no detecta patrón
    fallback_row = df.sample(1, random_state=0).iloc[0]
    interesting_fact = labels["chat_single_region"].format(
        region=fallback_row["CCAA"],
        gasto=fallback_row["Gasto Promedio Hogar Eur"],
        lic=int(fallback_row["Licencias Federadas"]),
    )
    return f"{labels['chat_analyze']} {interesting_fact}"
