"""
Central configuration for DEPORTEData.
"""

from __future__ import annotations

import os


USE_REAL_MODELS: bool = os.getenv("USE_REAL_MODELS", "true").lower() == "true"

QWEN_MODEL_DIR: str = os.getenv("QWEN_MODEL_DIR", "models/QwenBase")
QWEN_ADAPTER_DIR: str = os.getenv(
    "QWEN_ADAPTER_DIR",
    "models/QwenDeporteData/qwen2.5-finetuned/checkpoint-1443",
)
TOXICITY_MODEL_DIR: str = os.getenv(
    "TOXICITY_MODEL_DIR",
    "models/antiToxicidad/toxicity-classifier",
)

TOXICITY_THRESHOLD: float = float(os.getenv("TOXICITY_THRESHOLD", "0.82"))

PROCESSED_DIR: str = os.getenv("PROCESSED_DIR", "data/processed")
FEDERADOS_PARQUET: str = f"{PROCESSED_DIR}/federados.parquet"
GASTO_PARQUET: str = f"{PROCESSED_DIR}/gasto.parquet"

HF_QWEN_BASE_REPO: str = os.getenv("HF_QWEN_BASE_REPO", "Qwen/Qwen2.5-1.5B-Instruct")
HF_QWEN_REPO: str = os.getenv("HF_QWEN_REPO", "alfersal04/QwenDeporteData")
HF_TOXICITY_REPO: str = os.getenv("HF_TOXICITY_REPO", "alfersal04/antiToxicidad")

RAY_NUM_WORKERS: int = int(os.getenv("RAY_NUM_WORKERS", "4"))
RAY_USE_GPU: bool = os.getenv("RAY_USE_GPU", "true").lower() == "true"
