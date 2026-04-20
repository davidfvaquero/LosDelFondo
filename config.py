"""
config.py
=========
Configuración central de DEPORTEData.

Para activar los modelos de IA reales tras el entrenamiento:
  1. Entrena con: python scripts/ray_finetune.py --workers 4 --use-gpu
  2. Los modelos se suben automáticamente a Hugging Face Hub
  3. Cambia USE_REAL_MODELS = True

Todo lo demás sigue funcionando igual.
"""

# ─── Modelos de IA ─────────────────────────────────────────────────────────
# Cambia a True cuando los modelos estén entrenados y disponibles
USE_REAL_MODELS: bool = True

# Rutas locales (modelos guardados en disco tras el entrenamiento)
QWEN_MODEL_DIR:     str = "models/QwenBase"
QWEN_ADAPTER_DIR:   str = "models/QwenDeporteData/qwen2.5-finetuned/checkpoint-1443"
TOXICITY_MODEL_DIR: str = "models/antiToxicidad/toxicity-classifier"

# Umbral de toxicidad (0.0-1.0). Más alto = menos falsos positivos.
TOXICITY_THRESHOLD: float = 0.82

# ─── Datos ─────────────────────────────────────────────────────────────────
PROCESSED_DIR: str = "data/processed"
FEDERADOS_PARQUET: str = f"{PROCESSED_DIR}/federados.parquet"
GASTO_PARQUET:     str = f"{PROCESSED_DIR}/gasto.parquet"

# ─── Hugging Face Hub ──────────────────────────────────────────────────────
# Repositorios donde se publican los modelos entrenados con Ray Train.
# Actualizado automáticamente por scripts/ray_finetune.py al finalizar.
HF_USERNAME:      str = "alfersal"
HF_QWEN_REPO:     str = f"{HF_USERNAME}/qwen2.5-7b-deporte"
HF_TOXICITY_REPO: str = f"{HF_USERNAME}/toxicity-deporte-es"

# ─── Ray Training ──────────────────────────────────────────────────────────
# Configuración por defecto para el clúster Ray (ajustar según infraestructura)
RAY_NUM_WORKERS: int  = 4    # Número de workers GPU en el clúster
RAY_USE_GPU:     bool = True  # False para pruebas en CPU local
