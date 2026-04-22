#!/usr/bin/env python3
"""
aws/upload_models_to_s3.py
===========================
Sube los modelos locales al bucket S3 para que la EC2 los descargue en el arranque.

Uso:
    python aws/upload_models_to_s3.py

Variables de entorno: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
"""

import boto3
import os
import sys

S3_BUCKET  = os.environ.get("S3_BUCKET", "deportedata-models-data")
REGION     = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

s3 = boto3.client("s3", region_name=REGION)


def upload_folder(local_dir: str, s3_prefix: str):
    """Sube recursivamente una carpeta entera a S3."""
    if not os.path.exists(local_dir):
        print(f"  ⚠️  No existe: {local_dir} — saltando.")
        return
    total = 0
    for root, dirs, files in os.walk(local_dir):
        # Excluir carpetas pesadas innecesarias
        dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", "runs", "logs"]]
        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path   = os.path.relpath(local_path, BASE_DIR)
            s3_key     = rel_path.replace("\\", "/")
            size_mb    = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  ↑ {rel_path} ({size_mb:.1f} MB)...", end=" ", flush=True)
            s3.upload_file(local_path, S3_BUCKET, s3_key)
            print("✅")
            total += 1
    print(f"  → {total} archivos subidos desde {local_dir}")


def main():
    print("=" * 60)
    print(f"  DEPORTEData — Subiendo modelos a s3://{S3_BUCKET}")
    print("=" * 60)

    # 1. Parquets (obligatorio)
    print("\n📦 Datos (.parquet):")
    upload_folder(
        os.path.join(BASE_DIR, "data", "processed"),
        "data/processed"
    )

    # 2. Modelo de toxicidad
    print("\n🤖 Modelo Toxicidad:")
    upload_folder(
        os.path.join(BASE_DIR, "models", "antiToxicidad"),
        "models/antiToxicidad"
    )

    # 3. Modelo Qwen Base
    print("\n🤖 Qwen Base:")
    upload_folder(
        os.path.join(BASE_DIR, "models", "QwenBase"),
        "models/QwenBase"
    )

    # 4. Adaptador LoRA Fine-tuned
    print("\n🤖 Qwen LoRA Adapter:")
    upload_folder(
        os.path.join(BASE_DIR, "models", "QwenDeporteData"),
        "models/QwenDeporteData"
    )

    print("\n" + "=" * 60)
    print(f"  ✅ Subida completada a s3://{S3_BUCKET}")
    print("=" * 60)


if __name__ == "__main__":
    main()
