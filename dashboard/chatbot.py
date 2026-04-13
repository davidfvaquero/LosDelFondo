"""Pure chatbot helpers used by the Streamlit app and tests."""

from __future__ import annotations

import unicodedata

import pandas as pd
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

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

MAX_SPEND_PATTERNS = [
    "gasta mas",
    "maximo gasto",
    "most spending",
    "highest spending",
    "mas dinero",
    "gasto mas alto",
    "mayor gasto",
]
MIN_SPEND_PATTERNS = [
    "gasta menos",
    "minimo gasto",
    "least spending",
    "lowest spending",
    "menor gasto",
    "gasto mas bajo",
]
MAX_LICENSE_PATTERNS = [
    "mas licencias",
    "most licenses",
    "mas federados",
    "mas socios",
]

# Manual list of offensive/toxic terms to supplement the AI classifier.
# Includes common Spanish insults and slurs that may be missed by generic models.
MANUAL_TOXIC_TERMS = [
    "puta", "puto", "putos", "putas",
    "hijo de puta", "hija de puta",
    "inutil", "imbecil", "idiota", "gilipollas",
    "capullo", "mamona", "mamon", "culo",
    "mierda", "hostia", "joder", "coño",
    "pendejo", "cabron", "cabrona", "zorra",
    "fuck", "shit", "asshole", "bitch", "damn",
]


def normalize(text: str) -> str:
    """Return lowercase text without accents."""
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(char) != "Mn"
    )


def prepare_assistant_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names used by the chatbot."""
    return df.rename(
        columns={
            "Gasto_Promedio_Hogar_Eur": "Gasto Promedio Hogar Eur",
            "Licencias_Federadas": "Licencias Federadas",
        }
    )


def generate_chat_response(prompt: str, df: pd.DataFrame, labels: dict[str, str]) -> str:
    """Return a deterministic answer for the current prompt and dataset."""
    if df.empty:
        return labels["chat_error_data"]

    prompt_normalized = normalize(prompt)

    if any(pattern in prompt_normalized for pattern in MAX_SPEND_PATTERNS):
        row = df.loc[df["Gasto Promedio Hogar Eur"].idxmax()]
        return labels["chat_max_spend"].format(
            region=row["CCAA"],
            value=row["Gasto Promedio Hogar Eur"],
        )

    if any(pattern in prompt_normalized for pattern in MIN_SPEND_PATTERNS):
        row = df.loc[df["Gasto Promedio Hogar Eur"].idxmin()]
        return labels["chat_min_spend"].format(
            region=row["CCAA"],
            value=row["Gasto Promedio Hogar Eur"],
        )

    if any(pattern in prompt_normalized for pattern in MAX_LICENSE_PATTERNS):
        row = df.loc[df["Licencias Federadas"].idxmax()]
        return labels["chat_max_lic"].format(
            region=row["CCAA"],
            value=int(row["Licencias Federadas"]),
        )

    for alias, official_name in ALIASES.items():
        if alias in prompt_normalized:
            row = df[df["CCAA"] == official_name].iloc[0]
            return labels["chat_single_region"].format(
                region=official_name,
                gasto=row["Gasto Promedio Hogar Eur"],
                lic=int(row["Licencias Federadas"]),
            )

    fallback_row = df.sample(1, random_state=0).iloc[0]
    interesting_fact = labels["chat_single_region"].format(
        region=fallback_row["CCAA"],
        gasto=fallback_row["Gasto Promedio Hogar Eur"],
        lic=int(fallback_row["Licencias Federadas"]),
    )
    return f"{labels['chat_analyze']} {interesting_fact}"


# ── AI Models Integration ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelos de IA…")
def load_models():
    """Load the toxicity classifier and LLM pipeline (cached for the session)."""
    # 1. Toxicity Classifier
    toxic_tokenizer = AutoTokenizer.from_pretrained(
        "unitary/multilingual-toxic-xlm-roberta", use_fast=False
    )
    toxic_clf = pipeline(
        "text-classification",
        model="unitary/multilingual-toxic-xlm-roberta",
        tokenizer=toxic_tokenizer,
        top_k=None,
    )

    # 2. Causal LLM (Qwen2.5-0.5B-Instruct)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        device_map="auto",
    )
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        max_length=None,
        temperature=0.7,
        top_p=0.9,
    )

    return toxic_clf, llm_pipeline


def check_toxicity(prompt: str, classifier_pipeline) -> tuple[bool, float]:
    """Return (is_toxic, score).

    Two-layer check:
    1. Manual keyword list for common Spanish insults that generic models miss.
    2. AI classifier for everything else.
    """
    prompt_normalized = normalize(prompt)

    # Layer 1 – manual keyword guard
    for term in MANUAL_TOXIC_TERMS:
        if normalize(term) in prompt_normalized:
            return True, 1.0

    # Layer 2 – AI classifier
    try:
        results = classifier_pipeline(prompt)[0]
        for res in results:
            if res["label"] == "toxic":
                return (res["score"] > 0.5), res["score"]
        return False, 0.0
    except Exception as e:
        print(f"Toxicity check error: {e}")
        return False, 0.0


def build_dataset_context(df: pd.DataFrame) -> str:
    """Create a concise string representation of the DataFrame for the LLM."""
    if df.empty:
        return "No data available."

    lines = [
        "Dataset: gasto promedio por hogar en deporte (EUR) y licencias federadas por CCAA (España, 2023):"
    ]
    for _, row in df.iterrows():
        lines.append(
            f"- {row['CCAA']}: Gasto promedio {row['Gasto Promedio Hogar Eur']} EUR,"
            f" {int(row['Licencias Federadas'])} licencias federadas."
        )
    return "\n".join(lines)


def generate_llm_response(
    prompt: str, df: pd.DataFrame, llm_pipeline, lang: str
) -> str:
    """Generate a strictly data-grounded response using the Qwen model."""
    context = build_dataset_context(df)
    lang_name = "Spanish" if lang == "ES" else "English"

    system_msg = (
        f"You are DEPORTEData, a helpful data assistant for a Spanish sports analytics dashboard. "
        f"You MUST respond in {lang_name}.\n\n"
        "RULES (follow strictly):\n"
        "1. Answer ONLY questions about the dataset provided below.\n"
        "2. Do NOT invent, estimate or extrapolate data that is not in the dataset.\n"
        "3. If the user's question is rude, offensive, or completely unrelated to sports data, "
        "   politely decline and ask the user to please ask a relevant sports data question.\n"
        "4. If a question contains insults mixed with a real data question (e.g. 'dime que comunidad "
        "   gastó más inutil'), ignore the insult entirely and answer ONLY the data part.\n"
        "5. Be concise: answer in 1-3 sentences maximum.\n\n"
        f"Dataset context:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    output = llm_pipeline(messages)
    generated_text = output[0]["generated_text"][-1]["content"]
    return generated_text.strip()
