"""Pure chatbot helpers used by the Streamlit app and tests.

Estrategia de inferencia (por orden de prioridad):
  1. API local FastAPI en http://localhost:8000  ← preferido
  2. Modelos cargados directamente en proceso    ← si USE_DIRECT_MODELS=True
  3. Lógica determinista rule-based              ← siempre disponible como fallback
"""

from __future__ import annotations
import unicodedata
import os
import sys
import logging

import pandas as pd

log = logging.getLogger("deportedata.chatbot")

# ── Añadir raíz del proyecto al path ──────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Configuración central ──────────────────────────────────────────────────────
try:
    from config import (
        USE_REAL_MODELS,
        TOXICITY_MODEL_DIR,
        QWEN_MODEL_DIR,
        QWEN_ADAPTER_DIR,
        TOXICITY_THRESHOLD,
    )
except ImportError:
    USE_REAL_MODELS    = True
    TOXICITY_MODEL_DIR = "models/antiToxicidad/toxicity-classifier"
    QWEN_MODEL_DIR     = "models/QwenBase"
    QWEN_ADAPTER_DIR   = "models/QwenDeporteData/qwen2.5-finetuned/checkpoint-1443"
    TOXICITY_THRESHOLD = 0.82

# ── URL de la API local ────────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("DEPORTEDATA_API_URL", "http://localhost:8000")
API_TIMEOUT:  int = int(os.getenv("DEPORTEDATA_API_TIMEOUT", "120"))

# ── Diccionario de alias CCAA ──────────────────────────────────────────────────
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

# ── Patrones rule-based ────────────────────────────────────────────────────────
MAX_SPEND_PATTERNS    = ["gasta mas", "maximo gasto", "most spending", "highest spending", "mas dinero"]
MIN_SPEND_PATTERNS    = ["gasta menos", "minimo gasto", "least spending", "lowest spending"]
MAX_LICENSE_PATTERNS  = ["mas licencias", "most licenses", "mas federados", "mas socios"]


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  CLIENTE HTTP → API LOCAL
# ══════════════════════════════════════════════════════════════════════════════

def _api_health() -> bool:
    """Devuelve True si la API local está disponible y los modelos cargados."""
    try:
        import requests
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        data = resp.json()
        return resp.status_code == 200 and data.get("models_loaded", False)
    except Exception:
        return False


def _api_chat(prompt: str, lang: str = "ES") -> dict | None:
    """
    Llama a POST /chat de la API local.
    Devuelve el JSON de respuesta o None si falla.
    """
    try:
        import requests
        resp = requests.post(
            f"{API_BASE_URL}/chat",
            json={"prompt": prompt, "lang": lang},
            timeout=API_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()          # {"response": "...", "is_toxic": bool, "toxic_score": float}
        log.warning(f"API /chat devolvió status {resp.status_code}: {resp.text[:200]}")
        return None
    except Exception as exc:
        log.debug(f"API no disponible ({exc}); usando fallback.")
        return None


def _api_toxicity(text: str) -> dict | None:
    """
    Llama a POST /toxicity de la API local.
    Devuelve {"is_toxic": bool, "score": float} o None si falla.
    """
    try:
        import requests
        resp = requests.post(
            f"{API_BASE_URL}/toxicity",
            json={"text": text},
            timeout=10,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE MODELOS EN PROCESO (modo directo, sin API)
# ══════════════════════════════════════════════════════════════════════════════

def load_models():
    """
    Carga los modelos localmente dentro del proceso de Streamlit.
    Solo se usa si la API no está disponible y USE_REAL_MODELS=True.
    Usa @st.cache_resource cuando se llama desde Streamlit.
    """
    if not USE_REAL_MODELS:
        return None, None

    try:
        import torch
        import streamlit as st

        @st.cache_resource
        def _load():
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                AutoModelForCausalLM,
            )
            from peft import PeftModel

            # Toxicidad
            toxic_tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL_DIR)
            toxic_model = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL_DIR)
            toxic_model.eval()

            # Qwen + LoRA
            base_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_DIR)
            base_model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_DIR,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            base_model.eval()
            qwen_finetuned = PeftModel.from_pretrained(base_model, QWEN_ADAPTER_DIR)

            return (toxic_tokenizer, toxic_model), (base_tokenizer, qwen_finetuned)

        return _load()

    except Exception as exc:
        log.error(f"Error cargando modelos en proceso: {exc}")
        raise


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCIA DE TOXICIDAD
# ══════════════════════════════════════════════════════════════════════════════

def check_toxicity(text: str, classifier=None) -> tuple[bool, float]:
    """
    Verifica si el texto es tóxico.

    Prioridad:
      1. API local (si está disponible)
      2. Modelo en proceso (si classifier no es None)
      3. No tóxico por defecto
    """
    # Intentar vía API primero
    result = _api_toxicity(text)
    if result is not None:
        return result["is_toxic"], result["score"]

    # Fallback: modelo en proceso
    if classifier is not None:
        try:
            import torch
            toxic_tokenizer, toxic_model = classifier
            inputs = toxic_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = toxic_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            score    = probs[0][1].item()
            is_toxic = score > TOXICITY_THRESHOLD
            return is_toxic, score
        except Exception as exc:
            log.warning(f"check_toxicity (en proceso) error: {exc}")

    return False, 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE RESPUESTA
# ══════════════════════════════════════════════════════════════════════════════

def generate_llm_response(
    prompt: str,
    df: pd.DataFrame,
    pipeline=None,
    lang: str = "ES",
) -> str:
    """
    Genera una respuesta del chatbot con la siguiente prioridad:
      1. API local FastAPI   ← preferido
      2. Modelo en proceso   ← si pipeline no es None
      3. Lógica rule-based   ← siempre como último recurso
    """
    # ── 1. API local ──────────────────────────────────────────────────────────
    api_result = _api_chat(prompt, lang)
    if api_result is not None:
        if api_result.get("is_toxic"):
            blocked = (
                "⚠️ El mensaje ha sido bloqueado por nuestra política de seguridad debido a lenguaje tóxico."
                if lang == "ES"
                else "⚠️ The message has been blocked by our security policy due to toxic language."
            )
            return blocked
        response = api_result.get("response", "").strip()
        if response:
            return response

    # ── 2. Modelo en proceso ──────────────────────────────────────────────────
    if pipeline is not None:
        try:
            import torch
            qwen_tokenizer, qwen_model = pipeline

            system_prompt = (
                "Eres un asistente experto en deportes. Responde de forma directa, "
                "breve y profesional. Asegúrate de terminar la respuesta con un punto."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ]
            text   = qwen_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = qwen_tokenizer(text, return_tensors="pt").to(qwen_model.device)

            with torch.no_grad():
                output_ids = qwen_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=qwen_tokenizer.eos_token_id,
                )
            response_ids = output_ids[0][len(inputs.input_ids[0]):]
            response     = qwen_tokenizer.decode(response_ids, skip_special_tokens=True)
            if response.strip():
                return response.strip()
        except Exception as exc:
            log.warning(f"generate_llm_response (en proceso) error: {exc}")

    # ── 3. Fallback rule-based ─────────────────────────────────────────────────
    return _fallback_generate_chat_response(prompt, df, lang)


# ══════════════════════════════════════════════════════════════════════════════
#  LÓGICA DETERMINISTA (FALLBACK)
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_generate_chat_response(prompt: str, df: pd.DataFrame, lang: str) -> str:
    labels_map = {
        "ES": {
            "chat_max_spend":    "🔍 La CCAA que más gasta es {region} con {value} €.",
            "chat_min_spend":    "🔍 La CCAA que menos gasta es {region} con {value} €.",
            "chat_max_lic":      "🏆 {region} lidera en licencias con {value:,}.",
            "chat_single_region":"📍 En {region}: Gasto de {gasto} € y {lic} licencias.",
            "chat_analyze":      "🧠 He analizado los datos actuales:",
            "chat_error_data":   "⚠️ No hay datos disponibles para el análisis.",
        },
        "EN": {
            "chat_max_spend":    "🔍 The region with the highest spending is {region} with {value} €.",
            "chat_min_spend":    "🔍 The region with the lowest spending is {region} with {value} €.",
            "chat_max_lic":      "🏆 {region} leads in licenses with {value:,}.",
            "chat_single_region":"📍 In {region}: Spending of {gasto} € and {lic} licenses.",
            "chat_analyze":      "🧠 I have analyzed the current data:",
            "chat_error_data":   "⚠️ No data available for analysis.",
        },
    }

    labels = labels_map.get(lang, labels_map["ES"])

    if df.empty:
        return labels["chat_error_data"]

    prompt_normalized = normalize(prompt)

    if any(p in prompt_normalized for p in MAX_SPEND_PATTERNS):
        row = df.loc[df["Gasto Promedio Hogar Eur"].idxmax()]
        return labels["chat_max_spend"].format(region=row["CCAA"], value=row["Gasto Promedio Hogar Eur"])

    if any(p in prompt_normalized for p in MIN_SPEND_PATTERNS):
        row = df.loc[df["Gasto Promedio Hogar Eur"].idxmin()]
        return labels["chat_min_spend"].format(region=row["CCAA"], value=row["Gasto Promedio Hogar Eur"])

    if any(p in prompt_normalized for p in MAX_LICENSE_PATTERNS):
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
                    lic=int(row["Licencias Federadas"]),
                )

    fallback_row = df.sample(1, random_state=0).iloc[0]
    fact = labels["chat_single_region"].format(
        region=fallback_row["CCAA"],
        gasto=fallback_row["Gasto Promedio Hogar Eur"],
        lic=int(fallback_row["Licencias Federadas"]),
    )
    return f"{labels['chat_analyze']} {fact}"
