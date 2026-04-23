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
import pandas as pd
import numpy as np
import unicodedata
import re
import json
from datetime import datetime
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

# ── Logging (CloudWatch si está disponible) ────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("deportedata.api")

try:
    import watchtower, boto3
    _cw_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    _cw_handler = watchtower.CloudWatchLogHandler(
        log_group="/deportedata/api",
        stream_name="api-{strftime:%Y-%m-%d}",
        boto3_client=boto3.client("logs", region_name=_cw_region),
    )
    logging.getLogger().addHandler(_cw_handler)
    log.info("CloudWatch logging activado.")
except Exception:
    log.info("CloudWatch no disponible (entorno local). Usando logs locales.")

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

# ── Seguridad y Autenticación ────────────────────────────────────────────────
USERS_DB = {
    "admin": {
        "username": "admin",
        "password": "1234",
        "role": "administrator"
    }
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token.startswith("fake-token-"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username = token.replace("fake-token-", "")
    if username not in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return USERS_DB[username]

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
        t_path = TOXICITY_MODEL_DIR if os.path.exists(TOXICITY_MODEL_DIR) else "alfersal/qwen2.5-7b-deporte"
        log.info(f"  → Toxicity: {t_path} ({'local' if os.path.exists(TOXICITY_MODEL_DIR) else 'HuggingFace'})")
        state.toxic_tokenizer = AutoTokenizer.from_pretrained(t_path)
        state.toxic_model = AutoModelForSequenceClassification.from_pretrained(t_path)
        state.toxic_model.to(_device)
        state.toxic_model.eval()

        # 2. Cargar Qwen Base (Local -> HF Fallback)
        q_path = QWEN_MODEL_DIR if os.path.exists(QWEN_MODEL_DIR) else "alfersal/toxicity-deporte-est"
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
    user_ip: Optional[str] = None
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


# ── RDS Logger (con fallback a archivo local) ─────────────────────────────────
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from aws.rds_logger import log_chat, get_recent_logs, get_admin_stats as rds_admin_stats
    log.info("✅ Módulo RDS logger cargado correctamente.")
except ImportError:
    log.warning("⚠️ aws/rds_logger no encontrado. Usando log local de fallback.")
    CHAT_LOGS_FILE = "chat_logs.jsonl"

    def log_chat(ip: str, prompt: str, is_toxic: bool, toxic_score: float = 0.0,
                 response_length: int = 0, lang: str = "ES"):
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "ip": ip, "prompt": prompt, "is_toxic": is_toxic,
                "toxic_score": round(toxic_score, 4),
            }
            with open(CHAT_LOGS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error(f"Error saving chat log: {e}")

    def get_recent_logs(limit: int = 100):
        logs_list = []
        if os.path.exists(CHAT_LOGS_FILE):
            with open(CHAT_LOGS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try: logs_list.append(json.loads(line))
                    except: pass
        return list(reversed(logs_list))[-limit:]

    def rds_admin_stats(): return None

# ── Utilidades de Procesamiento de Datos ──────────────────────────────────────
def repair_mojibake(text: str) -> str:
    """Repara textos UTF-8 mal decodificados como latin-1 cuando es posible."""
    if pd.isna(text): return ""
    clean = str(text).replace("\ufeff", "").replace("ï»¿", "")
    try:
        repaired = clean.encode("latin1").decode("utf-8")
        bad_before = clean.count("Ã") + clean.count("ï")
        bad_after = repaired.count("Ã") + repaired.count("ï")
        if bad_after <= bad_before:
            clean = repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return clean.strip()

def normalize_column_name(name: str) -> str:
    """Normaliza nombres de columnas eliminando BOM, tildes, espacios y guiones."""
    clean = repair_mojibake(name).lower()
    clean = unicodedata.normalize("NFKD", clean)
    clean = "".join(c for c in clean if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-t0-9]', '', clean)

def normalize_for_match(text: str) -> str:
    """Normalización ultra-agresiva para cruzar datos (CCAA, Federaciones)."""
    clean = repair_mojibake(text).lower()
    clean = unicodedata.normalize("NFKD", clean)
    clean = "".join(c for c in clean if not unicodedata.combining(c))
    return re.sub(r'[^a-z]', '', clean)

def coalesce_normalized_columns(df: pd.DataFrame, canonical_map: dict[str, list[str]]) -> pd.DataFrame:
    """Crea columnas canónicas combinando variantes equivalentes por nombre."""
    result = df.copy()
    normalized_lookup: dict[str, list[str]] = {}
    for col in result.columns:
        normalized_lookup.setdefault(normalize_column_name(col), []).append(col)

    for canonical_name, aliases in canonical_map.items():
        candidate_cols: list[str] = []
        for alias in aliases:
            alias_key = normalize_column_name(alias)
            candidate_cols.extend(normalized_lookup.get(alias_key, []))
        
        # También buscar si el propio nombre canónico normalizado coincide con alguna columna
        can_key = normalize_column_name(canonical_name)
        candidate_cols.extend(normalized_lookup.get(can_key, []))
            
        if not candidate_cols:
            continue
        unique_candidates = list(dict.fromkeys(candidate_cols))
        result[canonical_name] = result[unique_candidates].bfill(axis=1).iloc[:, 0]

    return result

def parse_spanish_number(value):
    """Convierte números en formato español a float."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("\xa0", "").replace(" ", "")
    if not text or text == "..":
        return np.nan
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    elif text.count(".") == 1:
        left, right = text.split(".")
        if right.isdigit() and len(right) == 3 and left.isdigit():
            text = left + right
    try:
        return float(text)
    except ValueError:
        return np.nan

@lru_cache(maxsize=32)
def build_home_data(year: int) -> pd.DataFrame:
    """Carga y une federados.parquet + gasto.parquet para el año indicado."""
    log.info(f"📊 build_home_data({year}): Iniciando carga...")
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    fed_path = os.path.join(base_dir, "federados.parquet")
    gas_path = os.path.join(base_dir, "gasto.parquet")
    
    if not os.path.exists(fed_path) or not os.path.exists(gas_path):
        log.error(f"Archivos no encontrados: {fed_path} o {gas_path}")
        raise FileNotFoundError("Parquet files not found in data/processed/")

    EXCLUDED_CCAA_RAW = {'TOTAL', 'Sin territorializar', 'Ceuta', 'Melilla'}
    EXCLUDED_KEYS = {normalize_for_match(c) for c in EXCLUDED_CCAA_RAW}

    # 1. Cargar y filtrar por año inmediatamente para reducir memoria
    fed_raw = pd.read_parquet(fed_path)
    fed = fed_raw[pd.to_numeric(fed_raw['periodo'], errors='coerce') == int(year)].copy()
    
    gas_raw = pd.read_parquet(gas_path)
    gas = gas_raw[pd.to_numeric(gas_raw['periodo'], errors='coerce') == int(year)].copy()

    # 2. Coalescer columnas solo en el subconjunto
    fed = coalesce_normalized_columns(fed, {
        "federacion": ["Federación", "Federacion"],
        "comunidad_autonoma": ["Comunidad autónoma", "Comunidad autonoma"],
        "total_raw": ["Total"],
    })
    gas = coalesce_normalized_columns(gas, {
        "indicador": ["Indicador"],
        "comunidad_autonoma": ["Comunidad autónoma", "Comunidad autonoma"],
        "total_raw": ["Total"],
    })

    # 3. Procesar datos (mojibake, números, keys)
    for df in (fed, gas):
        if "total_raw" in df.columns:
            df["Total_Num"] = df["total_raw"].map(parse_spanish_number)
            
        for text_col in [col for col in ("federacion", "comunidad_autonoma", "indicador") if col in df.columns]:
            unique_vals = df[text_col].dropna().unique()
            val_map = {v: repair_mojibake(v) for v in unique_vals}
            df[text_col] = df[text_col].map(val_map)
            
        if "comunidad_autonoma" in df.columns:
            unique_ccaa = df["comunidad_autonoma"].dropna().unique()
            ccaa_map = {v: normalize_for_match(str(v)) for v in unique_ccaa}
            df["ccaa_key"] = df["comunidad_autonoma"].map(ccaa_map)

    # 4. Filtrar y Cruzar
    fed_year = (
        fed[
            (fed['archivo_origen'] == 'federado_01.csv')
            & (fed['federacion'].str.strip().str.upper() == 'TOTAL')
            & (~fed['ccaa_key'].isin(EXCLUDED_KEYS))
        ][['comunidad_autonoma', 'ccaa_key', 'Total_Num']]
        .rename(columns={'Total_Num': 'Licencias Federadas', 'comunidad_autonoma': 'CCAA'})
        .copy()
    )

    gas_year = (
        gas[
            (gas['archivo_origen'] == 'gasto_03.csv')
            & (gas['indicador'].str.contains('Gasto medio por hogar', na=False, case=False))
            & (gas['ccaa_key'] != normalize_for_match('TOTAL'))
        ][['ccaa_key', 'Total_Num']]
        .rename(columns={'Total_Num': 'Gasto Promedio Hogar Eur'})
    )

    merged_df = pd.merge(fed_year, gas_year, on='ccaa_key', how='inner')
    log.info(f"✅ build_home_data({year}) ok: {len(merged_df)} filas.")
    return merged_df.drop(columns=['ccaa_key'])

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
@app.get("/")
def read_root():
    return {"message": "Welcome to the DEPORTEData API"}


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


@app.post("/api/v1/token", tags=["Seguridad"])
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": f"fake-token-{user['username']}", "token_type": "bearer"}


@app.get("/api/v1/dashboard/metrics/{year}", tags=["Dashboard"])
def get_dashboard_metrics(year: int, territory: str = "Todas las CCAA"):
    try:
        df = build_home_data(year)
        if territory != "Todas las CCAA" and territory != "All Regions":
            df = df[df['CCAA'] == territory]
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Data for this year not found")

        return {
            "avg_spending": round(float(df['Gasto Promedio Hogar Eur'].mean()), 2),
            "total_licenses": int(df['Licencias Federadas'].sum()),
            "areas_analyzed": len(df)
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(f"Error in metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dashboard/charts/{year}", tags=["Dashboard"])
def get_dashboard_charts(year: int, territory: str = "Todas las CCAA"):
    try:
        df = build_home_data(year)
        if territory != "Todas las CCAA" and territory != "All Regions":
            df = df[df['CCAA'] == territory]
            
        if df.empty:
            raise HTTPException(status_code=404, detail="Data for this year not found")

        return df.to_dict(orient="records")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(f"Error in charts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dashboard/filters", tags=["Dashboard"])
def get_dashboard_filters():
    return {
        "years": [str(y) for y in range(2023, 2005, -1)],
        "territories": [
            "Todas las CCAA",
            "Andalucía", "Aragón", "Asturias, Principado de", "Balears, Illes",
            "Canarias", "Cantabria", "Castilla y León", "Castilla-La Mancha",
            "Cataluña", "Comunitat Valenciana", "Extremadura", "Galicia",
            "Madrid, Comunidad de", "Murcia, Región de", "Navarra, Comunidad Foral de",
            "País Vasco", "Rioja, La"
        ]
    }


@app.get("/api/v1/admin/stats", tags=["Admin"])
def get_admin_stats_endpoint(token: str = Depends(oauth2_scheme)):
    """Requiere autenticación Bearer. Retorna estadísticas reales desde RDS si disponible."""
    get_current_user(token)

    # Intentar obtener datos reales de RDS
    real_stats = rds_admin_stats()

    if real_stats:
        total = real_stats.get("total_queries", 0)
        daily = real_stats.get("daily_counts", [])
        return {
            "active_users": real_stats.get("active_users", 0),
            "total_queries": total,
            "chart_q": [d["count"] for d in daily],
            "chart_v": np.random.randn(len(daily)).tolist(),
            "total_by_day": [d["count"] for d in daily],
            "failed_attempts": real_stats.get("toxic_attempts", 0),
            "logs": get_recent_logs(10),
            "cpu_load": "N/A (CloudWatch)",
            "ram_usage": "N/A (CloudWatch)",
            "system_load": np.random.randn(20).tolist()
        }

    # Fallback con datos simulados si no hay RDS
    return {
        "active_users": 0,
        "total_queries": len(get_recent_logs(1000)),
        "chart_q": np.random.randn(20).tolist(),
        "chart_v": np.random.randn(20).tolist(),
        "total_by_day": np.random.randint(1, 20, size=7).tolist(),
        "failed_attempts": 0,
        "logs": get_recent_logs(10),
        "cpu_load": "N/A",
        "ram_usage": "N/A",
        "system_load": np.random.randn(20).tolist()
    }


@app.get("/api/v1/admin/chat_logs", tags=["Admin"])
def get_chat_logs(token: str = Depends(oauth2_scheme)):
    """Devuelve logs de chat desde RDS (o archivo local como fallback)."""
    get_current_user(token)
    return get_recent_logs(limit=200)


@app.post("/toxicity", response_model=ToxicityResponse, tags=["AI"])
def check_toxicity_endpoint(req: ToxicityRequest):
    """
    Comprueba si un texto es tóxico.

    - **text**: texto a analizar
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Modelos aún no cargados.")
    is_toxic, score = _check_toxicity(req.text)
    return ToxicityResponse(is_toxic=is_toxic, score=round(score, 4))


@app.post("/chat", response_model=ChatResponse, tags=["AI"])
def chat_endpoint(req: ChatRequest, request: Request):
    """
    Genera una respuesta del chatbot.

    Primero comprueba toxicidad; si el texto es tóxico devuelve `is_toxic=True`
    y una respuesta vacía. Si es limpio, pasa por el LLM Qwen finetuneado.

    - **prompt**: Pregunta del usuario
    - **lang**: "ES" o "EN"
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Modelos aún no cargados.")

    user_ip = req.user_ip or request.client.host

    # Layer 1: toxicidad
    is_toxic, toxic_score = _check_toxicity(req.prompt)
    if is_toxic:
        log_chat(user_ip, req.prompt, True, toxic_score=toxic_score, lang=req.lang)
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
        log_chat(user_ip, req.prompt, False,
                 toxic_score=toxic_score,
                 response_length=len(response),
                 lang=req.lang)
    except Exception as exc:
        log.error(f"Error en generación LLM: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        response=response,
        is_toxic=False,
        toxic_score=round(toxic_score, 4),
    )
