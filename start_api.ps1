# ════════════════════════════════════════════════════════════════
#  DEPORTEData — Arrancar API local de modelos IA (PowerShell)
#  Uso: .\start_api.ps1
# ════════════════════════════════════════════════════════════════

Set-Location $PSScriptRoot

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║   DEPORTEData  •  API local de Modelos IA   ║" -ForegroundColor Cyan
Write-Host "  ║   http://localhost:8000                      ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Activar entorno virtual si existe
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "[INFO] Activando entorno virtual .venv..." -ForegroundColor Yellow
    & ".venv\Scripts\Activate.ps1"
} else {
    Write-Host "[WARN] No se encontro .venv. Usando Python del sistema." -ForegroundColor Yellow
}

Write-Host "[INFO] Arrancando servidor FastAPI..." -ForegroundColor Green
Write-Host "[INFO] Los modelos pueden tardar 1-2 min en cargarse la primera vez." -ForegroundColor Green
Write-Host ""
Write-Host "  Pulsa Ctrl+C para detener." -ForegroundColor DarkGray
Write-Host ""

uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info
