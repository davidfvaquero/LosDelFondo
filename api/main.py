"""
api/main.py
===========
Servidor FastAPI que expone los modelos de IA locales (Qwen + toxicidad)
como endpoints HTTP. Se arranca una vez y los modelos permanecen en memoria.

Uso:
    cd <raíz del proyecto>
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health       — Estado del servidor y si los modelos están cargados
    POST /chat         — Genera respuesta del chatbot (toxicidad + LLM)
    POST /toxicity     — Solo comprueba si un texto es tóxico
"""

from __future__ import annotations

import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("deportedata.api")

# ── Importar configuración central ────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from config import (
        TOXICITY_MODEL_DIR,
        QWEN_MODEL_DIR,
        QWEN_ADAPTER_DIR,
        TOXICITY_THRESHOLD,
        HF_QWEN_REPO,
        HF_TOXICITY_REPO,
    )
except ImportError:
    TOXICITY_MODEL_DIR = "models/antiToxicidad/toxicity-classifier"
    QWEN_MODEL_DIR     = "models/QwenBase"
    QWEN_ADAPTER_DIR   = "models/QwenDeporteData/qwen2.5-finetuned/checkpoint-1443"
    TOXICITY_THRESHOLD = 0.82
    HF_QWEN_REPO       = "alfersal/qwen2.5-7b-deporte"
    HF_TOXICITY_REPO   = "alfersal/toxicity-deporte-es"

# ── Estado compartido de la app ────────────────────────────────────────────────
class ModelState:
    toxic_tokenizer = None
    toxic_model     = None
    qwen_tokenizer  = None
    qwen_model      = None
    loaded: bool    = False
    error: str      = ""

state = ModelState()


# ── Carga de modelos al arrancar ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos en el arranque; cierra recursos al apagar."""
    log.info("Cargando modelos de IA… esto puede tardar 1-2 min la primera vez.")
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            AutoModelForCausalLM,
        )
        from peft import PeftModel

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _dtype  = torch.float16 if _device == "cuda" else torch.float32

        # 1. Cargar Modelo de Toxicidad (Local -> HF Fallback)
        # Usamos un modelo público como fallback para asegurar que la conexión funcione sin tokens.
        t_path = TOXICITY_MODEL_DIR if os.path.exists(TOXICITY_MODEL_DIR) else "unitary/multilingual-toxic-xlm-roberta"
        log.info(f"  → Toxicity: {t_path} ({'local' if os.path.exists(TOXICITY_MODEL_DIR) else 'HuggingFace'})")
        state.toxic_tokenizer = AutoTokenizer.from_pretrained(t_path)
        state.toxic_model = AutoModelForSequenceClassification.from_pretrained(t_path)
        state.toxic_model.to(_device)
        state.toxic_model.eval()

        # 2. Cargar Qwen Base (Local -> HF Fallback)
        q_path = QWEN_MODEL_DIR if os.path.exists(QWEN_MODEL_DIR) else "Qwen/Qwen2.5-0.5B-Instruct"
        log.info(f"  → Qwen base: {q_path} ({'local' if os.path.exists(QWEN_MODEL_DIR) else 'HuggingFace'})")
        state.qwen_tokenizer = AutoTokenizer.from_pretrained(q_path)
        base = AutoModelForCausalLM.from_pretrained(
            q_path,
            dtype=_dtype,
            low_cpu_mem_usage=False,
        )
        
        # 3. Cargar Adaptador LoRA (Local -> HF Fallback)
        a_path = QWEN_ADAPTER_DIR if os.path.exists(QWEN_ADAPTER_DIR) else HF_QWEN_REPO
        log.info(f"  → LoRA adapter: {a_path} ({'local' if os.path.exists(QWEN_ADAPTER_DIR) else 'HuggingFace'})")
        
        base.to(_device)
        base.eval()
        
        try:
            state.qwen_model = PeftModel.from_pretrained(
                base,
                a_path,
                is_trainable=False,
            )
            log.info("  → Adaptador LoRA aplicado correctamente.")
        except Exception as lora_exc:
            log.warning(f"  ⚠️ No se pudo cargar el adaptador ({lora_exc}). Usando modelo base sin fine-tuning.")
            state.qwen_model = base

        state.loaded = True
        log.info("✅ Modelos cargados correctamente.")
    except Exception as exc:
        state.loaded = False
        state.error  = str(exc)
        log.error(f"❌ Error cargando modelos: {exc}")

    yield  # ── La app está corriendo ──

    log.info("Apagando servidor. Liberando modelos de memoria...")
    state.toxic_tokenizer = None
    state.toxic_model     = None
    state.qwen_tokenizer  = None
    state.qwen_model      = None


# ── Instancia FastAPI ──────────────────────────────────────────────────────────
app = FastAPI(
    title="DEPORTEData AI API",
    description="API local para inferencia de los modelos Qwen y toxicidad de DEPORTEData.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en producción, restringir a localhost
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas Pydantic ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str
    lang: str = "ES"
    max_new_tokens: int = 200
    temperature: float = 0.3
    top_p: float = 0.9
    repetition_penalty: float = 1.2

class ChatResponse(BaseModel):
    response: str
    is_toxic: bool
    toxic_score: float

class ToxicityRequest(BaseModel):
    text: str

class ToxicityResponse(BaseModel):
    is_toxic: bool
    score: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    device: str
    error: Optional[str] = None


# ── Helpers internos ───────────────────────────────────────────────────────────
def _check_toxicity(text: str) -> tuple[bool, float]:
    """Devuelve (is_toxic, score)."""
    if not state.loaded:
        return False, 0.0
    try:
        inputs = state.toxic_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = state.toxic_model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)
        score    = probs[0][1].item()
        is_toxic = score > TOXICITY_THRESHOLD
        return is_toxic, score
    except Exception as exc:
        log.warning(f"check_toxicity error: {exc}")
        return False, 0.0


def _generate(prompt: str, lang: str, max_new_tokens: int, temperature: float,
              top_p: float, repetition_penalty: float) -> str:
    """Genera respuesta con Qwen finetuneado."""
    if not state.loaded:
        raise RuntimeError("Modelos no cargados.")

    system_prompt = (
        "Eres un asistente experto en deportes. "
        "Responde de forma directa, breve y profesional. "
        "Asegúrate de terminar la respuesta con un punto."
    ) if lang == "ES" else (
        "You are an expert sports assistant. "
        "Answer directly, briefly and professionally. "
        "Make sure to end your response with a period."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]

    text   = state.qwen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = next(state.qwen_model.parameters()).device
    inputs = state.qwen_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = state.qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=state.qwen_tokenizer.eos_token_id,
        )

    response_ids = output_ids[0][len(inputs.input_ids[0]):]
    response     = state.qwen_tokenizer.decode(response_ids, skip_special_tokens=True)
    return response.strip() or "Lo siento, no sé cómo responder a eso."


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health():
    """Comprueba si el servidor y los modelos están listos."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HealthResponse(
        status="ok" if state.loaded else "degraded",
        models_loaded=state.loaded,
        device=device,
        error=state.error or None,
    )


@app.post("/toxicity", response_model=ToxicityResponse, tags=["Modelos"])
def check_toxicity_endpoint(req: ToxicityRequest):
    """
    Comprueba si un texto es tóxico.

    - **text**: texto a analizar
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Modelos aún no cargados.")
    is_toxic, score = _check_toxicity(req.text)
    return ToxicityResponse(is_toxic=is_toxic, score=round(score, 4))


@app.post("/chat", response_model=ChatResponse, tags=["Modelos"])
def chat_endpoint(req: ChatRequest):
    """
    Genera una respuesta del chatbot.

    Primero comprueba toxicidad; si el texto es tóxico devuelve `is_toxic=True`
    y una respuesta vacía. Si es limpio, pasa por el LLM Qwen finetuneado.

    - **prompt**: Pregunta del usuario
    - **lang**: "ES" o "EN"
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Modelos aún no cargados.")

    # Layer 1: toxicidad
    is_toxic, toxic_score = _check_toxicity(req.prompt)
    if is_toxic:
        return ChatResponse(
            response="",
            is_toxic=True,
            toxic_score=round(toxic_score, 4),
        )

    # Layer 2: generación LLM
    try:
        response = _generate(
            req.prompt,
            req.lang,
            req.max_new_tokens,
            req.temperature,
            req.top_p,
            req.repetition_penalty,
        )
    except Exception as exc:
        log.error(f"Error en generación LLM: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        response=response,
        is_toxic=False,
        toxic_score=round(toxic_score, 4),
    )
