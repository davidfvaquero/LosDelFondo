"""Pure chatbot helpers used by the Streamlit app and tests."""

from __future__ import annotations
import unicodedata
import pandas as pd


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
