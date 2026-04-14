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
    load_models, check_toxicity, generate_chat_response, 
    generate_llm_response, prepare_assistant_data
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
        'chat_max_spend': "🔍 La CCAA que más gasta es {region} con {value} €.",
        'chat_min_spend': "🔍 La CCAA que menos gasta es {region} con {value} €.",
        'chat_max_lic': "🏆 {region} lidera en licencias con {value:,}.",
        'chat_single_region': "📍 En {region}: Gasto de {gasto} € y {lic} licencias.",
        'chat_analyze': "🧠 He analizado los datos actuales. ¿Deseas algún detalle específico?",
        'chat_error_data': "⚠️ No hay datos disponibles para el análisis.",
        'chat_toxic_error': "⚠️ Mensaje detectado como inapropiado. Por favor, realiza consultas respetuosas."
    },
    'EN': {
        'chat_max_spend': "🔍 The region with the highest spending is {region} with {value} €.",
        'chat_min_spend': "🔍 The region with the lowest spending is {region} with {value} €.",
        'chat_max_lic': "🏆 {region} leads in licenses with {value:,}.",
        'chat_single_region': "📍 In {region}: Spending of {gasto} € and {lic} licenses.",
        'chat_analyze': "🧠 I have analyzed the current data. Do you need any specific details?",
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

@app.get("/api/v1/dashboard/metrics/{year}")
def get_dashboard_metrics(year: int, territory: str = "Todas las CCAA"):
    file_path = os.path.join(DATA_DIR, f"anio={year}", "hechos_indicadores.parquet")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Data for this year not found")
        
    try:
        df = pd.read_parquet(file_path)
        df = df.rename(columns={'Gasto_Promedio_Hogar_Eur': 'Gasto Promedio Hogar Eur', 'Licencias_Federadas': 'Licencias Federadas'})
        if territory != "Todas las CCAA" and territory != "All Regions":
            df = df[df['CCAA'] == territory]
        
        if df.empty:
            return {"avg_spending": 0, "total_licenses": 0, "areas_analyzed": 0}

        return {
            "avg_spending": df['Gasto Promedio Hogar Eur'].mean(),
            "total_licenses": int(df['Licencias Federadas'].sum()),
            "areas_analyzed": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/dashboard/charts/{year}")
def get_dashboard_charts(year: int, territory: str = "Todas las CCAA"):
    file_path = os.path.join(DATA_DIR, f"anio={year}", "hechos_indicadores.parquet")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Data for this year not found")
        
    try:
        df = pd.read_parquet(file_path)
        df = df.rename(columns={'Gasto_Promedio_Hogar_Eur': 'Gasto Promedio Hogar Eur', 'Licencias_Federadas': 'Licencias Federadas'})
        if territory != "Todas las CCAA" and territory != "All Regions":
            df = df[df['CCAA'] == territory]

        return df.to_dict(orient="records")
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
        file_path = os.path.join(DATA_DIR, "anio=2023", "hechos_indicadores.parquet")
        if not os.path.exists(file_path):
            return ChatResponse(response=L["chat_error_data"])
        
        df = pd.read_parquet(file_path)
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
