from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys

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

    fed = pd.read_parquet(fed_path)
    gas = pd.read_parquet(gas_path)

    # Licencias totales por CCAA (fila 'TOTAL' de federación)
    fed_year = (
        fed[
            (fed['periodo'] == year)
            & (fed['Federación'] == 'TOTAL')
            & (~fed['Comunidad autónoma'].isin(EXCLUDED_CCAA))
        ][['Comunidad autónoma', 'ccaa_limpia', 'Total_Num']]
        .rename(columns={'Total_Num': 'Licencias Federadas', 'Comunidad autónoma': 'CCAA'})
        .copy()
    )
    # Normalizar ccaa_limpia de federados para que coincida con gasto
    fed_year['ccaa_key'] = fed_year['ccaa_limpia'].map(
        lambda x: CCAA_KEY_MAP.get(x, x)
    )

    # Gasto medio por hogar
    gas_year = (
        gas[
            (gas['periodo'] == year)
            & (gas['Indicador'] == 'Gasto medio por hogar (Euros)')
            & (gas['Comunidad autónoma'] != 'TOTAL')
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
