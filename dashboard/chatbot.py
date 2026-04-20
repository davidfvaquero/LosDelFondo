"""Pure chatbot helpers used by the FastAPI backend."""

from __future__ import annotations
import unicodedata
import pandas as pd
# Configuración central — cambia USE_REAL_MODELS en config.py para activar la IA real
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
try:
    from config import USE_REAL_MODELS, TOXICITY_MODEL_DIR, QWEN_MODEL_DIR, TOXICITY_THRESHOLD
except ImportError:
    USE_REAL_MODELS    = False
    TOXICITY_MODEL_DIR = "models/toxicity-classifier"
    QWEN_MODEL_DIR     = "models/qwen2.5-7b-deporte"
    TOXICITY_THRESHOLD = 0.7
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

MANUAL_TOXIC_TERMS = [
    "tonto", "idiota", "estupido", "estúpido", "imbecil", "imbécil", 
    "subnormal", "gilipollas", "puta", "cabron", "cabrón", "mierda", "retrasado"
]

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

# ── AI Models Integration ────────────────────────────────────────────────────

def load_models():
    """Load the toxicity classifier and LLM pipeline."""
    import os
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification
    import peft

    # 1. Toxicity Classifier
    toxic_tokenizer = AutoTokenizer.from_pretrained(
        "unitary/multilingual-toxic-xlm-roberta", use_fast=False
    )
    tox_dir = snapshot_download(repo_id="alfersal04/antiToxicidad", allow_patterns="toxicity-classifier/*")
    tox_path = os.path.join(tox_dir, "toxicity-classifier")
    toxic_model = AutoModelForSequenceClassification.from_pretrained(tox_path)
    
    toxic_clf = pipeline(
        "text-classification",
        model=toxic_model,
        tokenizer=toxic_tokenizer,
        top_k=None,
    )

    # 2. Causal LLM (QwenDeporteData)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
    )
    
    qwen_dir = snapshot_download(repo_id="alfersal04/QwenDeporteData", allow_patterns="qwen2.5-finetuned/checkpoint-1443/*")
    adapter_path = os.path.join(qwen_dir, "qwen2.5-finetuned/checkpoint-1443")
    peft_model = peft.PeftModel.from_pretrained(model, adapter_path)
    
    llm_pipeline = pipeline(
        "text-generation",
        model=peft_model,
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
