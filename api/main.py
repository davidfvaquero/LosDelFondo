from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys

# Import test_chatbot module to reuse its normalize, aliases, and logic dependencies
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import test_chatbot

app = FastAPI(
    title="DEPORTEData API",
    description="API for accessing sports data statistics, analytics, and AI assistant",
    version="1.1.0"
)

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
        'chat_error_data': "⚠️ No hay datos disponibles para el análisis."
    },
    'EN': {
        'chat_max_spend': "🔍 The region with the highest spending is {region} with {value} €.",
        'chat_min_spend': "🔍 The region with the lowest spending is {region} with {value} €.",
        'chat_max_lic': "🏆 {region} leads in licenses with {value:,}.",
        'chat_single_region': "📍 In {region}: Spending of {gasto} € and {lic} licenses.",
        'chat_analyze': "🧠 I have analyzed the current data. Do you need any specific details?",
        'chat_error_data': "⚠️ No data available for analysis."
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
    Available to all users. Retains dependency on imported test_chatbot logic.
    """
    lang = request.lang if request.lang in LANGUAGES else "ES"
    L = LANGUAGES[lang]
    
    try:
        # Utilize the dataframe loaded by test_chatbot if available, else load manually map
        try:
            df_rag = test_chatbot.df
        except AttributeError:
            file_path = os.path.join(DATA_DIR, "anio=2023", "hechos_indicadores.parquet")
            df_rag = pd.read_parquet(file_path)
            df_rag = df_rag.rename(columns={'Gasto_Promedio_Hogar_Eur': 'Gasto Promedio Hogar Eur', 'Licencias_Federadas': 'Licencias Federadas'})
        
        # Use normalizer and aliases directly from test_chatbot
        p_low = test_chatbot.normalize(request.prompt)
        
        if any(x in p_low for x in ["gasta mas", "maximo gasto", "most spending", "highest spending", "mas dinero"]):
            row = df_rag.loc[df_rag['Gasto Promedio Hogar Eur'].idxmax()]
            resp = L['chat_max_spend'].format(region=row['CCAA'], value=row['Gasto Promedio Hogar Eur'])
        elif any(x in p_low for x in ["gasta menos", "minimo gasto", "least spending", "lowest spending"]):
            row = df_rag.loc[df_rag['Gasto Promedio Hogar Eur'].idxmin()]
            resp = L['chat_min_spend'].format(region=row['CCAA'], value=row['Gasto Promedio Hogar Eur'])
        elif any(x in p_low for x in ["mas licencias", "most licenses", "mas federados", "mas socios"]):
            row = df_rag.loc[df_rag['Licencias Federadas'].idxmax()]
            resp = L['chat_max_lic'].format(region=row['CCAA'], value=int(row['Licencias Federadas']))
        else:
            found = False
            for key, official_name in test_chatbot.aliases.items():
                if key in p_low:
                    row = df_rag[df_rag['CCAA'] == official_name].iloc[0]
                    resp = L['chat_single_region'].format(region=official_name, gasto=row['Gasto Promedio Hogar Eur'], lic=int(row['Licencias Federadas']))
                    found = True
                    break
            if not found:
                random_row = df_rag.sample(1).iloc[0]
                interesting_fact = L['chat_single_region'].format(region=random_row['CCAA'], gasto=random_row['Gasto Promedio Hogar Eur'], lic=int(random_row['Licencias Federadas']))
                resp = f"{L['chat_analyze']} {interesting_fact}"
                
        return ChatResponse(response=resp)
        
    except Exception as e:
        return ChatResponse(response=f"{L['chat_error_data']} ({str(e)})")
