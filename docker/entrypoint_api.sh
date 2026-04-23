#!/bin/bash
set -euo pipefail

echo "============================================"
echo "  DEPORTEData API - starting"
echo "============================================"

MODELS_DIR="/app/models"
DATA_DIR="/app/data/processed"
S3_BUCKET="${S3_BUCKET:-}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

mkdir -p "${MODELS_DIR}" "${DATA_DIR}"

sync_prefix() {
    local prefix="$1"
    local target="$2"
    if [ -z "${S3_BUCKET}" ]; then
        return 0
    fi

    mkdir -p "${target}"
    echo "[INFO] Syncing s3://${S3_BUCKET}/${prefix} -> ${target}"
    if aws s3 sync "s3://${S3_BUCKET}/${prefix}" "${target}" --region "${AWS_DEFAULT_REGION}" --only-show-errors; then
        echo "[OK] Sync complete for ${prefix}"
    else
        echo "[WARN] Could not sync ${prefix} from S3"
    fi
}

ensure_models() {
    local toxicity_dir="${MODELS_DIR}/antiToxicidad/toxicity-classifier"
    local qwen_base_dir="${MODELS_DIR}/QwenBase"
    local qwen_adapter_dir="${MODELS_DIR}/QwenDeporteData/qwen2.5-finetuned/checkpoint-1443"

    if [ ! -d "${toxicity_dir}" ] || [ ! -d "${qwen_base_dir}" ] || [ ! -d "${qwen_adapter_dir}" ]; then
        sync_prefix "models" "${MODELS_DIR}"
    fi

    if [ ! -d "${toxicity_dir}" ] || [ ! -d "${qwen_base_dir}" ] || [ ! -d "${qwen_adapter_dir}" ]; then
        echo "[INFO] Downloading models from Hugging Face ..."
        python /app/scripts/download_models.py
    fi
}

ensure_data() {
    if [ ! -d "${DATA_DIR}/federados.parquet" ] || [ ! -d "${DATA_DIR}/gasto.parquet" ]; then
        sync_prefix "data/processed" "${DATA_DIR}"
    fi
}

ensure_models
ensure_data

echo "[INFO] Starting FastAPI on 0.0.0.0:8000"
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --workers 1
