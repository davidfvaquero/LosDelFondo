"""Pure chatbot helpers used by the Streamlit app and tests."""

from __future__ import annotations
import os
import pandas as pd

# Fallback in case transformers or torch is not yet ready, we handle the import gracefully
try:
    from transformers import pipeline
    print("Loading toxicity classifier pipeline...")
    # top_k=None requests all class scores
    toxicity_classifier = pipeline("text-classification", model="unitary/multilingual-toxic-xlm-roberta", top_k=None)
    print("Classifier loaded successfully.")
except Exception as e:
    toxicity_classifier = None
    print(f"Error loading toxicity classifier: {e}")

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None


def prepare_assistant_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names used by the chatbot."""
    return df.rename(
        columns={
            "Gasto_Promedio_Hogar_Eur": "Gasto Promedio Hogar Eur",
            "Licencias_Federadas": "Licencias Federadas",
        }
    )

def is_toxic(prompt: str) -> bool:
    """Check if the prompt is considered toxic using the roberta model."""
    if not toxicity_classifier:
        print("Warning: toxicity classifier is not loaded, skipping check.")
        return False
    
    try:
        # Results is a list of lists when top_k=None on early pipelines, 
        # or just a list of dicts. We handle both:
        results = toxicity_classifier(prompt)
        
        # If it returns list of lists
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            scores = results[0]
        else:
            scores = results
            
        # Check probabilities of toxic labels.
        for label_score in scores:
            label = label_score['label']
            score = label_score['score']
            # unitary/multilingual-toxic-xlm-roberta uses labels like 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
            if label.lower() in ['toxic', 'insult', 'obscene', 'severe_toxic', 'threat', 'identity_hate'] and score > 0.6:
                return True
                
        return False
    except Exception as e:
        print(f"Error during toxicity classification: {e}")
        return False

def generate_chat_response(prompt: str, df: pd.DataFrame, hf_token: str | None = None) -> str:
    """
    Return a response from Mistral API based on the dataframe context.
    """
    if df.empty:
        return "Lo siento, no hay datos disponibles para procesar esta consulta."
        
    if is_toxic(prompt):
        return "⚠️ Tu mensaje parece contener lenguaje inapropiado y no puede ser procesado. Por favor, formula tu consulta con respeto."

    if InferenceClient is None:
        return "Error: la librería `huggingface_hub` no está instalada. Ejecuta `pip install huggingface_hub`."

    # Try to get Token from environment if not passed
    actual_token = hf_token or os.environ.get("HF_TOKEN")
    
    # We create a markdown representation of the dataframe to inject as context
    context = df.to_markdown(index=False)
    
    # Promt engineering for Mistral
    system_prompt = (
        "Eres el asistente IA de DEPORTEData. Eres un experto analista de datos amigable. "
        "Responde a la pregunta del usuario utilizando de forma EXCLUSIVA la información contenida en la siguiente tabla "
        "de datos sobre CCAA (Comunidad Autónoma), Gasto Promedio Hogar (en Euros) y Licencias Federadas. "
        "Intenta dar una respuesta concisa, profesional y en español o ingles, dependiendo en que idioma te pregunten,aclarando los datos correctos extraídos de la tabla. "
        "Limítate solo a responder la duda puntual basada en la tabla.\n\n"
        f"TABLA DE DATOS:\n{context}\n"
    )

    client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=actual_token)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=400,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_str = str(e)
        if "401" in error_str or "unauthorized" in error_str.lower():
            return "❌ Error de conexión: Se requiere un Token de HuggingFace (`HF_TOKEN`) válido con permisos de inferencia para acceder a Mistral."
        return f"❌ Ha ocurrido un error al conectarse a la API de Mistral: {error_str}"
