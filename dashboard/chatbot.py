"""Pure chatbot helpers used by the Streamlit app and tests."""

from __future__ import annotations
import unicodedata
import pandas as pd
import streamlit as st
import torch
import os

import sys

# Agregar la raíz del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuración central
try:
    from config import USE_REAL_MODELS, TOXICITY_MODEL_DIR, QWEN_MODEL_DIR, QWEN_ADAPTER_DIR, TOXICITY_THRESHOLD
except ImportError:
    USE_REAL_MODELS    = True
    TOXICITY_MODEL_DIR = "models/antiToxicidad/toxicity-classifier"
    QWEN_MODEL_DIR     = "models/QwenBase"
    QWEN_ADAPTER_DIR   = "models/QwenDeporteData/qwen2.5-finetuned/checkpoint-1443"
    TOXICITY_THRESHOLD = 0.82

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

# --- CARGA DE MODELOS CON CACHÉ ---
@st.cache_resource
def load_models():
    """
    Carga los modelos de IA utilizando memoria caché para que sobrevivan a los recargos de página.
    """
    if not USE_REAL_MODELS:
        return None, None

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
        from peft import PeftModel
        
        # 1. Cargar modelo de toxicidad
        toxic_tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL_DIR)
        toxic_model = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL_DIR)
        toxic_model.eval()
        toxic_clf = (toxic_tokenizer, toxic_model)
        
        # 2. Cargar modelo Qwen base y aplicar adaptador LoRA
        base_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_DIR)
        base_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        base_model.eval()
        qwen_finetuned = PeftModel.from_pretrained(base_model, QWEN_ADAPTER_DIR)
        llm_pipeline = (base_tokenizer, qwen_finetuned)
        
        return toxic_clf, llm_pipeline
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        # Retorna el error para diagnosticar de inmediato desde la GUI
        raise e

# --- INTEGRACIÓN ML (TOXICIDAD) ---
def check_toxicity(text: str, classifier=None):
    """
    Verifica si el texto es tóxico con base en TOXICITY_THRESHOLD (0.82)
    """
    if classifier is None:
        return False, 0.0

    try:
        toxic_tokenizer, toxic_model = classifier
        inputs = toxic_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = toxic_model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            
        toxic_prob = predictions[0][1].item()
        is_toxic = toxic_prob > TOXICITY_THRESHOLD
        return is_toxic, toxic_prob
    except Exception as e:
        print(f"Error en check_toxicity: {e}")
        return False, 0.0

# --- INTEGRACIÓN ML (QWEN) ---
def generate_llm_response(prompt: str, df: pd.DataFrame, pipeline=None, lang: str="ES") -> str:
    """
    Punto de entrada que conecta el prompt con la lógica de datos real o al LLM Finetuned con RAG.
    """
    if pipeline is None:
        # Fallback a la lógica de RAG simple por si fallan los modelos
        return _fallback_generate_chat_response(prompt, df, lang)
    
    qwen_tokenizer, qwen_model = pipeline
    
    system_prompt = "Eres un asistente experto en deportes. Responde de forma directa, breve y profesional. Asegúrate de terminar la respuesta con un punto."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tokenizer(text, return_tensors="pt").to(qwen_model.device)
    
    with torch.no_grad():
        output_ids = qwen_model.generate(
            **inputs, 
            max_new_tokens=200, 
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=qwen_tokenizer.eos_token_id
        )
        response_ids = output_ids[0][len(inputs.input_ids[0]):]
        response = qwen_tokenizer.decode(response_ids, skip_special_tokens=True)
        
    return response if response.strip() else "Lo siento, no sé cómo responder a eso."

# --- LÓGICA DE RESPUESTA (FALLBACK SI MODELOS ESTÁN APAGADOS) ---
def _fallback_generate_chat_response(prompt: str, df: pd.DataFrame, lang: str) -> str:
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
    
    labels = labels_map.get(lang, labels_map["ES"])
    
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
            match = df[df["CCAA"] == official_name]
            if not match.empty:
                row = match.iloc[0]
                return labels["chat_single_region"].format(
                    region=official_name,
                    gasto=row["Gasto Promedio Hogar Eur"],
                    lic=int(row["Licencias Federadas"])
                )

    fallback_row = df.sample(1, random_state=0).iloc[0]
    interesting_fact = labels["chat_single_region"].format(
        region=fallback_row["CCAA"],
        gasto=fallback_row["Gasto Promedio Hogar Eur"],
        lic=int(fallback_row["Licencias Federadas"]),
    )
    return f"{labels['chat_analyze']} {interesting_fact}"
