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
]
MIN_SPEND_PATTERNS = [
    "gasta menos",
    "minimo gasto",
    "least spending",
    "lowest spending",
]
MAX_LICENSE_PATTERNS = [
    "mas licencias",
    "most licenses",
    "mas federados",
    "mas socios",
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

# AI Models Integration

@st.cache_resource(show_spinner="Loading NLP models...")
def load_models():
    """Load the toxicity classifier and LLM pipeline."""
    # 1. Toxicity Classifier
    toxic_tokenizer = AutoTokenizer.from_pretrained("unitary/multilingual-toxic-xlm-roberta", use_fast=False)
    toxic_clf = pipeline(
        "text-classification",
        model="unitary/multilingual-toxic-xlm-roberta",
        tokenizer=toxic_tokenizer,
        top_k=None
    )

    # 2. Causual LLM (Qwen2.5)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        device_map="auto"
    )
    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        max_length=None,
        temperature=0.7,
        top_p=0.9
    )

    return toxic_clf, llm_pipeline

def check_toxicity(prompt: str, classifier_pipeline) -> tuple[bool, float]:
    """Return True if prompt is toxic, along with the toxicity score."""
    try:
        results = classifier_pipeline(prompt)[0]
        # Get the 'toxic' label score
        for res in results:
            if res['label'] == 'toxic':
                return (res['score'] > 0.5), res['score']
        return False, 0.0
    except Exception as e:
        print(f"Toxicity check error: {e}")
        return False, 0.0

def build_dataset_context(df: pd.DataFrame) -> str:
    """Create a string representation of the Dataframe for the LLM."""
    if df.empty:
        return "No data available."
    
    # Just serialize the top data rows or agg
    context = "Dataset Summary (avg spend in EUR and federated licenses by region):\n"
    for _, row in df.iterrows():
        context += f"- {row['CCAA']}: Avg Spend {row['Gasto Promedio Hogar Eur']} EUR, {int(row['Licencias Federadas'])} licenses.\n"
    return context

def generate_llm_response(prompt: str, df: pd.DataFrame, llm_pipeline, lang: str) -> str:
    """Generate response using Qwen model with the dataframe as context."""
    context = build_dataset_context(df)
    
    sys_msg = (
        "You are an AI assistant for DEPORTEData, an analytics dashboard about sports spending and licenses in Spain. "
        "Answer the user's question concisely based ONLY on the provided dataset context. If the data does not contain the answer, say so. "
        f"Respond in {'Spanish' if lang == 'ES' else 'English'}."
    )
    
    messages = [
        {"role": "system", "content": sys_msg + "\n\nContext:\n" + context},
        {"role": "user", "content": prompt}
    ]
    
    output = llm_pipeline(messages)
    generated_text = output[0]["generated_text"][-1]["content"]
    return generated_text.strip()

