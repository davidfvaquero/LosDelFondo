@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
echo [INFO] Iniciando API en http://localhost:8000...
uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info
pause
