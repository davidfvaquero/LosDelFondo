#!/usr/bin/env python3
"""
aws/deploy.py
==============
Script maestro de despliegue completo de DEPORTEData en AWS.
Ejecuta todos los pasos en orden:
  1. Construye las imágenes Docker y las sube a Docker Hub
  2. Crea la infraestructura AWS (S3, RDS, EC2, CloudWatch)
  3. Sube los modelos y parquets a S3
  4. Espera a que la EC2 esté lista y verifica la salud de la API

Uso:
    python aws/deploy.py
"""

import subprocess
import sys
import os
import json
import time
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCKER_HUB_USER  = os.environ.get("DOCKERHUB_USERNAME", "nabreue01")
DOCKER_HUB_TOKEN = os.environ.get("DOCKERHUB_TOKEN", "")


def run(cmd: list[str], cwd: str = BASE_DIR) -> int:
    """Ejecuta un comando mostrando output en tiempo real."""
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def step(n: int, total: int, title: str):
    print(f"\n{'='*60}")
    print(f"  PASO {n}/{total}: {title}")
    print(f"{'='*60}")


def check_api_health(ip: str, max_retries: int = 20, delay: int = 30) -> bool:
    """Espera a que la API responda en /health."""
    url = f"http://{ip}:8000/health"
    print(f"\n  ⏳ Esperando API en {url}...")
    for i in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = json.loads(r.read())
                if data.get("status") == "ok":
                    print(f"  ✅ API lista! Modelos cargados: {data.get('models_loaded')}")
                    return True
                else:
                    print(f"  [{i+1}/{max_retries}] Status: {data.get('status')} — esperando...")
        except Exception as e:
            print(f"  [{i+1}/{max_retries}] No disponible aún ({type(e).__name__})...")
        time.sleep(delay)
    return False


def main():
    print("\n" + "="*60)
    print("  DEPORTEData — DESPLIEGUE COMPLETO EN AWS")
    print("="*60)

    # Cargar .env
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        print("  📄 Cargando variables desde .env...")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    # ── PASO 1: Login Docker Hub ──────────────────────────────────────────────
    step(1, 5, "Login a Docker Hub")
    print("  ✅ Login de Docker realizado previamente vía CLI.")

    # ── PASO 2: Build y Push imágenes Docker ─────────────────────────────────
    step(2, 5, "Build & Push Docker Images")

    images = [
        ("Dockerfile.api",       f"{DOCKER_HUB_USER}/deportedata-api:latest"),
        ("Dockerfile.dashboard", f"{DOCKER_HUB_USER}/deportedata-dashboard:latest"),
    ]
    for dockerfile, tag in images:
        print(f"\n  🐳 Building {tag}...")
        ret = run(["docker", "build", "-f", dockerfile, "-t", tag, "."])
        if ret != 0:
            print(f"  ❌ Build falló para {tag}")
            sys.exit(1)
        print(f"  ⬆️  Pushing {tag}...")
        ret = run(["docker", "push", tag])
        if ret != 0:
            print(f"  ❌ Push falló para {tag}")
            sys.exit(1)
        print(f"  ✅ {tag} publicada en Docker Hub.")

    # ── PASO 3: Infraestructura AWS ───────────────────────────────────────────
    step(3, 5, "Crear Infraestructura AWS (S3 + RDS + EC2 + CloudWatch)")
    ret = run([sys.executable, "aws/setup_infrastructure.py"])
    if ret != 0:
        print("  ❌ Error creando infraestructura.")
        sys.exit(1)

    # Leer IP pública generada
    info_path = os.path.join(BASE_DIR, "aws", "deployment_info.json")
    if not os.path.exists(info_path):
        print("  ❌ No se encontró aws/deployment_info.json")
        sys.exit(1)
    with open(info_path) as f:
        info = json.load(f)
    public_ip = info["public_ip"]

    # ── PASO 4: Subir modelos a S3 ────────────────────────────────────────────
    step(4, 5, "Subir modelos y parquets a S3")
    ret = run([sys.executable, "aws/upload_models_to_s3.py"])
    if ret != 0:
        print("  ⚠️  Algunos modelos no se pudieron subir. La EC2 los descargará de HuggingFace.")

    # ── PASO 5: Verificar despliegue ──────────────────────────────────────────
    step(5, 5, "Verificar despliegue")
    print("\n  ⏳ Esperando que la EC2 configure Docker y arranque los contenedores...")
    print("  (Esto puede tardar 5-10 minutos en la primera ejecución)")
    time.sleep(60)

    api_ok = check_api_health(public_ip, max_retries=15, delay=30)

    print("\n" + "="*60)
    print("  ✅ DESPLIEGUE COMPLETADO")
    print("="*60)
    print(f"  🌐 API:       http://{public_ip}:8000")
    print(f"  🌐 Dashboard: http://{public_ip}:8501")
    print(f"  📊 API Docs:  http://{public_ip}:8000/docs")
    if not api_ok:
        print("\n  ⚠️  La API aún no responde. Los modelos pueden seguir cargándose.")
        print(f"     Comprueba el estado en: http://{public_ip}:8000/health")
    print("="*60)


if __name__ == "__main__":
    main()
