#!/bin/bash
# ── entrypoint_api.sh ──────────────────────────────────────────────────────
# Entrypoint del contenedor API:
#   1. Descarga modelos desde S3 si no existen localmente
#   2. Arranca uvicorn
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "============================================"
echo "  DEPORTEData API - Iniciando..."
echo "============================================"

# Variables de entorno esperadas (inyectadas desde EC2 userdata / docker-compose):
#   S3_BUCKET         — Nombre del bucket S3
#   AWS_DEFAULT_REGION — Región de AWS
#   DB_HOST, DB_NAME, DB_USER, DB_PASSWORD — Para logs RDS

MODELS_DIR="/app/models"
S3_BUCKET="${S3_BUCKET:-deportedata-models}"

# ── 1. Descargar modelos desde S3 si no están presentes ──────────────────
download_from_s3() {
    local s3_path="$1"
    local local_path="$2"
    if [ ! -d "$local_path" ]; then
        echo "[INFO] Descargando $s3_path desde S3..."
        mkdir -p "$local_path"
        aws s3 sync "s3://${S3_BUCKET}/${s3_path}" "$local_path" --quiet
        echo "[OK] $local_path listo."
    else
        echo "[INFO] $local_path ya existe, saltando descarga."
    fi
}

# Descargar modelos desde S3
download_from_s3 "models/QwenBase"          "${MODELS_DIR}/QwenBase"
download_from_s3 "models/QwenDeporteData"   "${MODELS_DIR}/QwenDeporteData"
download_from_s3 "models/antiToxicidad"     "${MODELS_DIR}/antiToxicidad"

# ── 2. Descargar parquets desde S3 si no están ───────────────────────────
DATA_DIR="/app/data/processed"
mkdir -p "$DATA_DIR"

for file in federados.parquet gasto.parquet; do
    if [ ! -f "${DATA_DIR}/${file}" ]; then
        echo "[INFO] Descargando ${file} desde S3..."
        aws s3 cp "s3://${S3_BUCKET}/data/processed/${file}" "${DATA_DIR}/${file}"
        echo "[OK] ${file} listo."
    fi
done

# ── 3. Arrancar API ──────────────────────────────────────────────────────
echo ""
echo "[INFO] Arrancando FastAPI (uvicorn)..."
echo "[INFO] API disponible en http://0.0.0.0:8000"
echo ""

exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --workers 1
