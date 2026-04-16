"""
prepare_finetuning_data.py
==========================
Genera automáticamente un dataset de fine-tuning en formato JSONL
partir de todos los CSV crudos en data/raw/.

Cada registro tiene el formato:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Ejecutar desde la raíz del proyecto:
    python scripts/prepare_finetuning_data.py
"""

from __future__ import annotations

import glob
import json
import os
import random
import re

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
RAW_DIR = "data/raw"
OUTPUT_PATH = "data/processed/train_dataset.jsonl"
MAX_SAMPLES_PER_FILE = 40   # filas por CSV para no sobreajustar
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "Eres un experto analista de estadísticas deportivas de España. "
    "Tienes acceso a datos del Ministerio de Educación, FP y Deporte sobre "
    "deporte federado, gasto en deporte, hábitos deportivos, empleo en el sector "
    "y empresas deportivas. Respondes siempre en español de forma clara y precisa."
)

random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina BOM e espacios en nombres de columna y en strings."""
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    # Limpiar celdas string también
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def safe_num(val: str) -> str:
    """Formatea el valor Total (puede venir con puntos de miles españoles)."""
    val = str(val).strip().replace(".", "").replace(",", ".")
    try:
        n = float(val)
        if n.is_integer():
            return f"{int(n):,}".replace(",", ".")
        return f"{n:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except ValueError:
        return val


def make_pair(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# GENERADORES DE PARES POR TIPO DE DATO
# ─────────────────────────────────────────────────────────────────────────────

def pairs_from_federado(df: pd.DataFrame, filename: str) -> list[dict]:
    pairs = []
    has_ccaa   = "Comunidad autónoma" in df.columns
    has_fed    = "Federación" in df.columns
    has_sexo   = "Sexo" in df.columns

    sample = df.dropna(subset=["Total"]).sample(
        min(MAX_SAMPLES_PER_FILE, len(df)), random_state=RANDOM_SEED
    )

    for _, row in sample.iterrows():
        total = safe_num(row["Total"])
        periodo = row.get("periodo", "N/D")
        ccaa    = row.get("Comunidad autónoma", "") if has_ccaa else ""
        fed     = row.get("Federación", "") if has_fed else ""
        sexo    = row.get("Sexo", "") if has_sexo else ""

        # Patrón 1: consulta directa
        partes_q = [f"deportistas de {fed}" if fed else "deportistas federados"]
        partes_a = ["deportistas" if not fed else f"deportistas de la federación {fed}"]
        if has_sexo and sexo:
            partes_q[0] += f" ({sexo.lower()})"
        if has_ccaa and ccaa:
            partes_q.append(f"en {ccaa}")
            partes_a.append(f"en {ccaa}")
        partes_q.append(f"en {periodo}")
        partes_a.append(f"en el periodo {periodo} fue de {total}")

        q = f"¿Cuántos {' '.join(partes_q)}?"
        a = f"El número de {' '.join(partes_a)}."
        pairs.append(make_pair(q, a))

    return pairs


def pairs_from_gasto(df: pd.DataFrame, filename: str) -> list[dict]:
    pairs = []
    has_ccaa   = "Comunidad autónoma" in df.columns
    has_ind    = "Indicador" in df.columns
    has_tipo   = "Tipo de bienes y servicios" in df.columns

    sample = df.dropna(subset=["Total"]).sample(
        min(MAX_SAMPLES_PER_FILE, len(df)), random_state=RANDOM_SEED
    )

    for _, row in sample.iterrows():
        total  = safe_num(row["Total"])
        periodo = row.get("periodo", "N/D")
        indicador = row.get("Indicador", "gasto deportivo") if has_ind else "gasto deportivo"
        ccaa      = row.get("Comunidad autónoma", "") if has_ccaa else ""
        tipo      = row.get("Tipo de bienes y servicios", "") if has_tipo else ""

        q_parts = [f"¿Cuál fue el {indicador.lower()}"]
        a_parts = [f"El {indicador.lower()}"]
        if ccaa:
            q_parts.append(f"en {ccaa}")
            a_parts.append(f"en {ccaa}")
        if tipo:
            q_parts.append(f"para '{tipo}'")
            a_parts.append(f"para '{tipo}'")
        q_parts.append(f"en {periodo}?")
        a_parts.append(f"en {periodo} fue de {total}.")

        pairs.append(make_pair(" ".join(q_parts), " ".join(a_parts)))

    return pairs


def pairs_from_empleo(df: pd.DataFrame, filename: str) -> list[dict]:
    pairs = []
    sample = df.dropna(subset=["Total"]).sample(
        min(MAX_SAMPLES_PER_FILE, len(df)), random_state=RANDOM_SEED
    )

    for _, row in sample.iterrows():
        total     = safe_num(row["Total"])
        periodo   = row.get("periodo", row.get("Periodo", "N/D"))
        indicador = row.get("Indicador", "empleo deportivo")
        perfil    = row.get("Sexo, grupo de edad y nivel de estudios", "")

        q = f"¿Cuál fue el valor de '{indicador}'{' para ' + perfil if perfil else ''} en {periodo}?"
        a = f"El indicador '{indicador}'{' para el perfil ' + perfil if perfil else ''} en {periodo} registró un valor de {total}."
        pairs.append(make_pair(q, a))

    return pairs


def pairs_from_empresa(df: pd.DataFrame, filename: str) -> list[dict]:
    pairs = []
    sample = df.dropna(subset=["Total"]).sample(
        min(MAX_SAMPLES_PER_FILE, len(df)), random_state=RANDOM_SEED
    )

    for _, row in sample.iterrows():
        total   = safe_num(row["Total"])
        periodo = row.get("periodo", "N/D")
        ind     = row.get("Indicador", "empresas deportivas")
        act     = row.get("Actividad económica", "")

        q = f"¿Qué valor tuvo '{ind}'{' en la actividad ' + act if act else ''} en {periodo}?"
        a = f"El indicador '{ind}'{' para la actividad económica ' + act if act else ''} en {periodo} fue de {total}."
        pairs.append(make_pair(q, a))

    return pairs


def pairs_from_habitos(df: pd.DataFrame, filename: str) -> list[dict]:
    pairs = []
    sample = df.dropna(subset=["Total"]).sample(
        min(MAX_SAMPLES_PER_FILE, len(df)), random_state=RANDOM_SEED
    )

    # Inferir año de la ruta (columna no siempre presente)
    year_match = re.search(r"(20\d{2})", filename)
    anio = year_match.group(1) if year_match else "N/D"

    for _, row in sample.iterrows():
        total   = safe_num(row["Total"])
        zona    = str(row.get("Comunidad autónoma y tamaño del municipio", "España")).strip()
        ind     = str(row.get("Indicador", "hábito deportivo")).strip()

        q = f"¿Cuál fue el porcentaje de '{ind}' en {zona} (encuesta {anio})?"
        a = f"Según la encuesta de hábitos deportivos de {anio}, el indicador '{ind}' en {zona} fue del {total}%."
        pairs.append(make_pair(q, a))

    return pairs


def pairs_from_afiliacion(df: pd.DataFrame, filename: str) -> list[dict]:
    pairs = []
    sample = df.dropna(subset=["Total"]).sample(
        min(MAX_SAMPLES_PER_FILE, len(df)), random_state=RANDOM_SEED
    )

    for _, row in sample.iterrows():
        total   = safe_num(row["Total"])
        periodo = row.get("Periodo", "N/D")
        tipo    = row.get("Tipo de indicador", "afiliación")
        sexo    = row.get("Sexo", "")
        act     = row.get("Actividad económica", "")

        q = f"¿Cuál fue el indicador de '{tipo}' para {sexo + ', ' if sexo else ''}actividad '{act}' en {periodo}?"
        a = f"El indicador '{tipo}' {('para ' + sexo + ' ') if sexo else ''}en {periodo} fue de {total}."
        pairs.append(make_pair(q, a))

    return pairs


def pairs_from_eu(df: pd.DataFrame, filename: str) -> list[dict]:
    pairs = []
    sample = df.dropna(subset=["Total"]).sample(
        min(MAX_SAMPLES_PER_FILE, len(df)), random_state=RANDOM_SEED
    )

    for _, row in sample.iterrows():
        total   = safe_num(row["Total"])
        periodo = row.get("periodo", "N/D")
        pais    = str(row.get("País", "N/D")).strip()
        ind     = row.get("Indicador", "indicador EU")
        perfil  = row.get("Sexo y nivel de estudios", "")

        q = f"¿Cuál fue el valor de '{ind}' en {pais}{', ' + perfil if perfil else ''} ({periodo})?"
        a = f"El indicador armonizado '{ind}' en {pais}{' para ' + perfil if perfil else ''} en {periodo} fue de {total}."
        pairs.append(make_pair(q, a))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER: asigna generador según carpeta padre
# ─────────────────────────────────────────────────────────────────────────────

DIR_ROUTER = {
    "deporte_federado": pairs_from_federado,
    "empleo_deporte":   pairs_from_empleo,
    "empresa_deporte":  pairs_from_empresa,
    "gasto_hogares":    pairs_from_gasto,
    "gasto_publico":    pairs_from_gasto,
    "habitos_deportivos": pairs_from_habitos,
    "indicadores_afiliacion": pairs_from_afiliacion,
    "indicadore_armonizacion_eu": pairs_from_eu,
}


def detect_category(filepath: str) -> str:
    """Devuelve la categoría basándose en la primera subcarpeta de data/raw."""
    parts = filepath.replace("\\", "/").lower().split("/")
    raw_idx = next((i for i, p in enumerate(parts) if p == "raw"), -1)
    if raw_idx != -1 and raw_idx + 1 < len(parts):
        return parts[raw_idx + 1]
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# AÑADIR CONOCIMIENTO GENERAL DEL PROYECTO
# ─────────────────────────────────────────────────────────────────────────────

GENERAL_QA = [
    # ── Identidad y presentación ──────────────────────────────────────────────
    make_pair(
        "¿Quién eres?",
        "Soy el asistente IA de DEPORTEData, un sistema de análisis inteligente especializado "
        "en estadísticas deportivas de España. Puedo ayudarte a consultar datos sobre federaciones "
        "deportivas, gasto de los hogares en deporte, hábitos deportivos, empleo y empresas del sector. "
        "¡Pregúntame lo que necesites!"
    ),
    make_pair(
        "¿Cómo te llamas?",
        "Soy el asistente de DEPORTEData. Estoy aquí para ayudarte a explorar y entender las "
        "estadísticas deportivas de España de forma sencilla y conversacional."
    ),
    make_pair(
        "¿Para qué sirves?",
        "Sirvo para responder preguntas sobre el deporte en España usando datos reales. Puedes "
        "preguntarme cuántas licencias tiene una federación, cómo ha evolucionado el gasto en "
        "deporte por hogar, qué comunidad autónoma practica más deporte, cómo está el empleo en "
        "el sector o cómo nos comparamos con otros países europeos, entre muchas otras cosas."
    ),
    make_pair(
        "Hola",
        "¡Hola! Soy el asistente de DEPORTEData. Estoy listo para ayudarte con cualquier consulta "
        "sobre estadísticas deportivas en España. ¿En qué puedo ayudarte hoy?"
    ),
    make_pair(
        "Buenos días",
        "¡Buenos días! ¿En qué puedo ayudarte hoy? Puedo resolver dudas sobre federaciones "
        "deportivas, gasto en deporte, hábitos de la población o empleo en el sector deportivo."
    ),
    make_pair(
        "Gracias",
        "¡De nada! Si tienes más preguntas sobre los datos deportivos de España, aquí estaré. "
        "¡Hasta pronto!"
    ),
    make_pair(
        "Adiós",
        "¡Hasta luego! Ha sido un placer ayudarte. Si necesitas más información sobre el deporte "
        "en España, vuelve cuando quieras."
    ),

    # ── Proyecto DEPORTEData ───────────────────────────────────────────────────
    make_pair(
        "¿Qué es el proyecto DEPORTEData?",
        "DEPORTEData es una plataforma de análisis e inteligencia deportiva que estudia "
        "la evolución del deporte en España. Integra datos de múltiples fuentes oficiales "
        "sobre federaciones deportivas, gasto de los hogares, hábitos deportivos de la "
        "población, empleo en el sector y empresas deportivas."
    ),
    make_pair(
        "¿Qué fuentes de datos utiliza DEPORTEData?",
        "DEPORTEData utiliza datos de varias fuentes oficiales: el Ministerio de Educación, "
        "FP y Deporte (licencias federadas y estadísticas de clubes), el INE mediante la "
        "Encuesta de Presupuestos Familiares (gasto por hogar en deporte), las Encuestas de "
        "Hábitos Deportivos, la Seguridad Social (afiliaciones al sector) y las estadísticas "
        "de empresas deportivas. También incluye indicadores armonizados a nivel europeo."
    ),
    make_pair(
        "¿A qué periodo temporal cubren los datos?",
        "Los datos disponibles cubren principalmente desde el año 2005 hasta 2024-2025, "
        "permitiendo análisis de tendencias de casi dos décadas en el deporte español."
    ),
    make_pair(
        "¿Quién ha desarrollado DEPORTEData?",
        "DEPORTEData ha sido desarrollado como proyecto académico por un equipo universitario "
        "dentro del marco del Reto A de Inteligencia Artificial aplicada al sector deportivo."
    ),
    make_pair(
        "¿Qué tecnologías usa DEPORTEData?",
        "DEPORTEData combina varias tecnologías: Python para el procesamiento de datos, "
        "Streamlit para el dashboard interactivo, modelos de lenguaje de la familia Qwen "
        "con fine-tuning específico para el dominio deportivo, y un sistema RAG (Retrieval "
        "Augmented Generation) para responder preguntas con información actualizada."
    ),
    make_pair(
        "¿Cómo puedo ver los gráficos de los datos?",
        "Puedes acceder a los gráficos e indicadores interactivos a través del dashboard "
        "de DEPORTEData, disponible en la interfaz principal de la aplicación Streamlit. "
        "Desde allí podrás explorar la evolución de licencias federadas, gasto y hábitos "
        "deportivos por comunidad autónoma y año."
    ),

    # ── Conocimiento del dominio deportivo ────────────────────────────────────
    make_pair(
        "¿Qué son las licencias deportivas federadas?",
        "Las licencias deportivas federadas son carnets oficiales expedidos por las federaciones "
        "deportivas españolas que acreditan a un deportista para competir de forma oficial. "
        "Son un indicador clave de la práctica deportiva organizada en España."
    ),
    make_pair(
        "¿Cuál es el deporte con más licencias federadas en España?",
        "Históricamente, el fútbol es el deporte con mayor número de licencias federadas en "
        "España, seguido de deportes como el baloncesto, el tenis, el atletismo y las artes "
        "marciales. Los datos exactos pueden variar según el año consultado."
    ),
    make_pair(
        "¿Qué comunidad autónoma tiene más deportistas federados?",
        "En términos absolutos, Cataluña, Madrid y Andalucía suelen encabezar el ranking de "
        "deportistas federados al ser las comunidades más pobladas. Sin embargo, si se analiza "
        "por tasa relativa de federados sobre la población, comunidades más pequeñas como "
        "Navarra o el País Vasco suelen destacar."
    ),
    make_pair(
        "¿Ha aumentado el gasto en deporte de los hogares españoles?",
        "Sí, en términos generales el gasto de los hogares españoles en deporte ha mostrado una "
        "tendencia creciente desde 2005, con algunas caídas puntuales como la provocada por la "
        "crisis financiera de 2008-2012 y la pandemia de COVID-19 en 2020. Los datos de la "
        "Encuesta de Presupuestos Familiares del INE reflejan esta evolución."
    ),
    make_pair(
        "¿Qué indica la encuesta de hábitos deportivos?",
        "La Encuesta de Hábitos Deportivos mide el porcentaje de la población española que "
        "practica deporte, con qué frecuencia, qué deportes practica, sus motivaciones y las "
        "barreras para la práctica. Se realiza periódicamente y permite comparar la evolución "
        "de la cultura deportiva en España a lo largo del tiempo."
    ),
    make_pair(
        "¿Cuántas personas trabajan en el sector deportivo en España?",
        "El empleo en el sector deportivo en España ha crecido de forma sostenida. Los datos "
        "de afiliación a la Seguridad Social muestran las personas empleadas en actividades "
        "deportivas, recreativas y de entretenimiento, diferenciando por sexo, grupo de edad "
        "y tipo de actividad económica."
    ),
    make_pair(
        "¿Qué son los indicadores armonizados de la UE en deporte?",
        "Los indicadores armonizados de la Unión Europea permiten comparar la práctica deportiva "
        "entre distintos países miembros utilizando la misma metodología. Incluyen datos como el "
        "porcentaje de población que practica deporte regularmente, diferenciado por sexo y nivel "
        "educativo, lo que permite situar a España en el contexto europeo."
    ),
    make_pair(
        "¿Cómo está España en comparación con Europa en práctica deportiva?",
        "Según los indicadores armonizados de la UE, España se sitúa por debajo de la media "
        "europea en práctica deportiva regular, aunque ha mejorado en las últimas décadas. "
        "Los países escandinavos y centroeuropeos suelen liderar los rankings de actividad "
        "física de la población."
    ),

    # ── Límites y casos que el modelo no debe responder ──────────────────────
    make_pair(
        "¿Cuánto vale el Barça?",
        "Lo siento, pero ese tipo de información está fuera de mi ámbito. Estoy especializado "
        "en estadísticas oficiales del sector deportivo español: licencias federadas, gasto de "
        "los hogares, hábitos deportivos, empleo y empresas del sector. ¿Te puedo ayudar con "
        "alguna de estas áreas?"
    ),
    make_pair(
        "¿Quién ganó el último mundial de fútbol?",
        "Esa pregunta está fuera de mi especialidad. Soy un asistente de análisis de datos "
        "estadísticos del deporte en España, no un asistente de noticias deportivas. Para "
        "resultados de competiciones te recomiendo consultar fuentes como la FIFA o la RFEF."
    ),
    make_pair(
        "¿Puedes escribirme un correo electrónico?",
        "No, eso no está dentro de mis capacidades. Soy un asistente especializado en "
        "estadísticas deportivas de España. Puedo ayudarte a analizar datos de federaciones, "
        "gasto, hábitos deportivos o empleo en el sector. ¿En qué te puedo ayudar?"
    ),
    make_pair(
        "No sé qué preguntarte",
        "¡No te preocupes! Aquí tienes algunas ideas: puedes preguntarme cuántos deportistas "
        "federados hay en tu comunidad autónoma, cómo ha evolucionado el gasto en deporte de "
        "los hogares, qué porcentaje de españoles practica deporte regularmente, o cómo se "
        "compara España con otros países europeos en actividad física. ¿Por cuál empezamos?"
    ),
    make_pair(
        "No entiendo los datos, ¿me lo puedes explicar de forma más sencilla?",
        "Por supuesto, es para eso que estoy aquí. Dime el dato o concepto que quieres entender "
        "mejor y te lo explico de forma clara y sin tecnicismos. Los datos deportivos pueden "
        "parecer complejos al principio, pero juntos los hacemos fáciles de interpretar."
    ),

    # ── Preguntas técnicas sobre el propio sistema ────────────────────────────
    make_pair(
        "¿Cómo funciona el asistente de DEPORTEData?",
        "El asistente combina dos técnicas de inteligencia artificial: un modelo de lenguaje "
        "fine-tuneado con datos deportivos reales para entender el dominio, y un sistema RAG "
        "(Recuperación Aumentada por Generación) que consulta los datos más actualizados antes "
        "de responder. Así puedo darte respuestas precisas basadas en información real."
    ),
    make_pair(
        "¿Los datos que usas son fiables?",
        "Sí. Todos los datos provienen de fuentes oficiales del gobierno español: el Ministerio "
        "de Educación, FP y Deporte, el Instituto Nacional de Estadística (INE) y la Seguridad "
        "Social, entre otros. Son los mismos datos que usan investigadores y administraciones "
        "públicas para tomar decisiones sobre política deportiva."
    ),
    make_pair(
        "¿Puedes equivocarte en tus respuestas?",
        "Como cualquier sistema de inteligencia artificial, puedo cometer errores, especialmente "
        "con datos muy específicos o preguntas muy complejas. Siempre te recomiendo contrastar "
        "los datos importantes con las fuentes originales. Si algo no te cuadra, ¡dímelo y lo "
        "revisamos juntos!"
    ),
    make_pair(
        "¿Con qué frecuencia se actualizan los datos?",
        "Los datos se actualizan según las publicaciones oficiales de cada fuente. Las licencias "
        "federadas se publican anualmente por el Ministerio de Deporte, la Encuesta de Presupuestos "
        "Familiares la actualiza el INE anualmente, y los datos de Hábitos Deportivos se publican "
        "cada ciertos años. Los datos más recientes disponibles en el sistema llegan hasta 2024-2025."
    ),

    # ── Preguntas comparativas y de tendencias ─────────────────────────────────
    make_pair(
        "¿Ha aumentado el deporte federado en España en los últimos años?",
        "Sí, el número de licencias deportivas federadas en España ha crecido de forma general "
        "a lo largo de los últimos 20 años, aunque con variaciones según el deporte y la comunidad "
        "autónoma. La pandemia de 2020 provocó una bajada notable, pero los datos posteriores "
        "muestran una clara recuperación y crecimiento."
    ),
    make_pair(
        "¿Practica más deporte la gente joven o la mayor?",
        "Según las encuestas de hábitos deportivos, la práctica deportiva es significativamente "
        "mayor entre la población más joven. A medida que aumenta la edad, el porcentaje de "
        "personas que practican deporte regularmente tiende a disminuir, aunque en los últimos años "
        "se observa un crecimiento de la actividad física entre los mayores de 55 años."
    ),
    make_pair(
        "¿Hay diferencias en la práctica deportiva entre hombres y mujeres?",
        "Sí, históricamente los hombres han tenido tasas de práctica deportiva y federación más "
        "altas que las mujeres en España. Sin embargo, la brecha se ha ido reduciendo progresivamente "
        "en las últimas décadas, con un crecimiento notable de la participación femenina en el deporte "
        "federado y en la práctica deportiva general."
    ),
    make_pair(
        "¿Qué comunidad autónoma gasta más en deporte por hogar?",
        "Los datos de la Encuesta de Presupuestos Familiares muestran variaciones importantes entre "
        "comunidades autónomas. En general, las comunidades con mayor renta per cápita tienden a "
        "registrar un mayor gasto en deporte por hogar. Para conocer el ranking exacto de un año "
        "concreto, puedo consultarlo en los datos disponibles."
    ),
    make_pair(
        "¿Cuántos clubes deportivos hay en España?",
        "Los datos del Ministerio de Deporte recogen el número de clubes deportivos federados en "
        "España para cada año. El número total varía según el deporte y la comunidad autónoma, "
        "y ha mostrado una tendencia creciente a lo largo de los años. Si me indicas un año "
        "o deporte concreto, puedo darte el dato específico."
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    all_pairs: list[dict] = []

    # Añadir QA general primero
    all_pairs.extend(GENERAL_QA)

    # Recorrer todos los CSV recursivamente
    csv_files = glob.glob(os.path.join(RAW_DIR, "**", "*.csv"), recursive=True)
    print(f"[INFO] Encontrados {len(csv_files)} archivos CSV en {RAW_DIR}")

    for filepath in sorted(csv_files):
        category = detect_category(filepath)
        generator = DIR_ROUTER.get(category)

        if not generator:
            print(f"  [SKIP] Sin generador para categoria: '{category}' -> {os.path.basename(filepath)}")
            continue

        try:
            df = pd.read_csv(filepath, sep=";", encoding="latin1")
            df = clean_columns(df)

            if "Total" not in df.columns:
                print(f"  [SKIP] Sin columna 'Total': {os.path.basename(filepath)}")
                continue

            pairs = generator(df, filepath)
            all_pairs.extend(pairs)
            print(f"  [OK] {os.path.basename(filepath)} → {len(pairs)} pares (cat: {category})")

        except Exception as exc:
            print(f"  [ERROR] {os.path.basename(filepath)}: {exc}")

    # Mezclar aleatoriamente para evitar order-bias
    random.shuffle(all_pairs)

    # Escribir JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in all_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[✓] Dataset generado con {len(all_pairs)} ejemplos → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
