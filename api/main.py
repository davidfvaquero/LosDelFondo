from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys
import unicodedata
from functools import lru_cache

# Import components from the new structural location
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dashboard.chatbot import (
    load_models, check_toxicity, generate_llm_response, prepare_assistant_data
)

app = FastAPI(
    title="DEPORTEData API",
    description="API for accessing sports data statistics, analytics, and AI assistant",
    version="1.1.0"
)

# --- AI Models Globals ---
toxic_clf = None
llm_pipeline = None

# --- Authentication Configuration ---
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

# --- Data Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "deporte_data")

# --- Models ---
class ChatRequest(BaseModel):
    prompt: str
    lang: str = "ES"

class ChatResponse(BaseModel):
    response: str

# --- Translation Dictionary (for AI) ---
LANGUAGES = {
    'ES': {
        'chat_error_data': "⚠️ No hay datos disponibles para el análisis.",
        'chat_toxic_error': "⚠️ Mensaje detectado como inapropiado. Por favor, realiza consultas respetuosas."
    },
    'EN': {
        'chat_error_data': "⚠️ No data available for analysis.",
        'chat_toxic_error': "⚠️ Message detected as inappropriate. Please make respectful queries."
    }
}

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the DEPORTEData API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/v1/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": f"fake-token-{user['username']}", "token_type": "bearer"}

# --- Data Helper ---
def repair_mojibake(text: str) -> str:
    """Repara textos UTF-8 mal decodificados como latin-1 cuando es posible."""
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
    """Normaliza nombres de columnas eliminando BOM, tildes y ruido de encoding."""
    clean = repair_mojibake(name).lower()
    clean = unicodedata.normalize("NFKD", clean)
    clean = "".join(c for c in clean if not unicodedata.combining(c))
    return clean

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
            for normalized_col, original_cols in normalized_lookup.items():
                if normalized_col.endswith(alias_key) and normalized_col != alias_key:
                    candidate_cols.extend(original_cols)
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
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    fed_path = os.path.join(base_dir, "federados.parquet")
    gas_path = os.path.join(base_dir, "gasto.parquet")
    
    if not os.path.exists(fed_path) or not os.path.exists(gas_path):
        raise FileNotFoundError("Raw parquet files not found")

    EXCLUDED_CCAA = {'TOTAL', 'Sin territorializar', 'Ceuta', 'Melilla'}

    # Mapeo ccaa_limpia de federados → clave equivalente en gasto
    CCAA_KEY_MAP = {
        'asturias, principado de':     'asturias (principado de)',
        'balears, illes':              'balears (illes)',
        'madrid, comunidad de':        'madrid (comunidad de)',
        'murcia, regi\u00f3n de':           'murcia (regi\u00f3n de)',
        'navarra, comunidad foral de': 'navarra (comunidad foral de)',
        'rioja, la':                   'rioja (la)',
    }

    fed = coalesce_normalized_columns(
        pd.read_parquet(fed_path),
        {
            "federacion": ["Federación", "Federacion"],
            "comunidad_autonoma": ["Comunidad autónoma", "Comunidad autonoma"],
            "periodo": ["periodo"],
            "total_raw": ["Total"],
        },
    )
    gas = coalesce_normalized_columns(
        pd.read_parquet(gas_path),
        {
            "indicador": ["Indicador"],
            "comunidad_autonoma": ["Comunidad autónoma", "Comunidad autonoma"],
            "periodo": ["periodo"],
            "total_raw": ["Total"],
        },
    )
    yr = int(year)

    for df in (fed, gas):
        df["periodo"] = pd.to_numeric(df["periodo"], errors="coerce")
        
        if "total_raw" in df.columns:
            unique_totals = df["total_raw"].unique()
            tot_map = {val: parse_spanish_number(val) for val in unique_totals}
            df["Total_Num"] = df["total_raw"].map(tot_map)
            
        for text_col in [col for col in ("federacion", "comunidad_autonoma", "indicador") if col in df.columns]:
            unique_texts = df[text_col].dropna().unique()
            text_map = {val: repair_mojibake(val) for val in unique_texts}
            df[text_col] = df[text_col].map(lambda x: text_map.get(x, x))
            
        if "comunidad_autonoma" in df.columns:
            unique_ccaa = df["comunidad_autonoma"].dropna().unique()
            ccaa_map = {val: normalize_column_name(str(val)) for val in unique_ccaa}
            df["ccaa_limpia"] = df["comunidad_autonoma"].map(lambda x: ccaa_map.get(x, x))

    # Licencias totales por CCAA (fuente federado_01, fila 'TOTAL' de federación)
    fed_year = (
        fed[
            (fed['periodo'] == yr)
            & (fed['archivo_origen'] == 'federado_01.csv')
            & (fed['federacion'] == 'TOTAL')
            & (~fed['comunidad_autonoma'].isin(EXCLUDED_CCAA))
        ][['comunidad_autonoma', 'ccaa_limpia', 'Total_Num']]
        .rename(columns={'Total_Num': 'Licencias Federadas', 'comunidad_autonoma': 'CCAA'})
        .copy()
    )
    # Normalizar ccaa_limpia de federados para que coincida con gasto
    fed_year['ccaa_key'] = fed_year['ccaa_limpia'].map(
        lambda x: CCAA_KEY_MAP.get(x, x)
    )

    # Gasto medio por hogar
    gas_year = (
        gas[
            (gas['periodo'] == yr)
            & (gas['archivo_origen'] == 'gasto_03.csv')
            & (gas['indicador'] == 'Gasto medio por hogar (Euros)')
            & (gas['comunidad_autonoma'] != 'TOTAL')
        ][['ccaa_limpia', 'Total_Num']]
        .rename(columns={'Total_Num': 'Gasto Promedio Hogar Eur', 'ccaa_limpia': 'ccaa_key'})
    )

    df = pd.merge(fed_year, gas_year, on='ccaa_key', how='inner')
    df = df.drop(columns=['ccaa_limpia', 'ccaa_key'])
    return df


@app.get("/api/v1/dashboard/metrics/{year}")
def get_dashboard_metrics(year: int, territory: str = "Todas las CCAA"):
    try:
        df = build_home_data(year)
        if territory != "Todas las CCAA" and territory != "All Regions":
            df = df[df['CCAA'] == territory]
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Data for this year not found")

        return {
            "avg_spending": df['Gasto Promedio Hogar Eur'].mean(),
            "total_licenses": int(df['Licencias Federadas'].sum()),
            "areas_analyzed": len(df)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Raw data files not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/dashboard/charts/{year}")
def get_dashboard_charts(year: int, territory: str = "Todas las CCAA"):
    try:
        df = build_home_data(year)
        if territory != "Todas las CCAA" and territory != "All Regions":
            df = df[df['CCAA'] == territory]
            
        if df.empty:
            raise HTTPException(status_code=404, detail="Data for this year not found")

        return df.to_dict(orient="records")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Raw data files not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/dashboard/filters")
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

@app.get("/api/v1/admin/stats")
def get_admin_stats(token: str = Depends(oauth2_scheme)):
    # Simulates admin backend logic
    get_current_user(token)
    return {
        "active_users": 142,
        "total_queries": 2840,
        "chart_q": np.random.randn(20).tolist(),
        "chart_v": np.random.randn(20).tolist(),
        "total_by_day": np.random.randint(10, 100, size=7).tolist(),
        "failed_attempts": 3,
        "logs": [
            {"User": "admin", "IP": "192.168.1.45", "Status": "Success"},
            {"User": "root", "IP": "85.23.11.102", "Status": "Blocked"},
            {"User": "guest", "IP": "172.16.0.5", "Status": "Failed"},
            {"User": "admin", "IP": "192.168.1.45", "Status": "Success"}
        ],
        "cpu_load": "24%",
        "ram_usage": "1.2 GB",
        "system_load": np.random.randn(20).tolist()
    }

@app.post("/api/v1/chat", response_model=ChatResponse)
def ai_chat_assistant(request: ChatRequest):
    """
    AI Chat endpoint handling specific queries about sports data.
    Accommodates the latest refactors in dashboard/chatbot.py.
    """
    global toxic_clf, llm_pipeline
    
    lang = request.lang if request.lang in LANGUAGES else "ES"
    L = LANGUAGES[lang]
    
    try:
        # 1. Load Data
        try:
            df = build_home_data(2023)
        except Exception:
            return ChatResponse(response=L["chat_error_data"])
        
        if df.empty:
            return ChatResponse(response=L["chat_error_data"])
            
        df_rag = prepare_assistant_data(df)
        
        # 2. Lazy load HF models (toxicity & LLM)
        if toxic_clf is None or llm_pipeline is None:
            toxic_clf, llm_pipeline = load_models()
            
        # 3. Check Toxicity
        is_toxic, _ = check_toxicity(request.prompt, toxic_clf)
        if is_toxic:
            return ChatResponse(response=L["chat_toxic_error"])
            
        # 4. Generate dynamic AI response using the LLM for all queries
        resp = generate_llm_response(request.prompt, df_rag, llm_pipeline, lang)
            
        return ChatResponse(response=resp)
        
    except Exception as e:
        return ChatResponse(response=f"{L['chat_error_data']} ({str(e)})")
