# LosDelFondo

Proyecto con una web/dashboard en Streamlit para explorar indicadores de gasto deportivo y licencias federadas.

## Enlace de la web

Aplicacion desplegada: https://losdelfondo-frontend.onrender.com/

## CI antes de desplegar en Render

Se ha anadido el workflow de GitHub Actions en [`.github/workflows/ci.yml`](/LosDelFondo/.github/workflows/ci.yml) para validar el proyecto antes del despliegue.

Checks que ejecuta:

- instala dependencias
- ejecuta lint con Ruff
- valida que los archivos Python compilan
- ejecuta tests con pytest
- regenera el dataset procesado
- comprueba que el Parquet existe y tiene la estructura esperada

## CD hacia Render

Se ha anadido el workflow [`.github/workflows/render-deploy.yml`](/LosDelFondo/.github/workflows/render-deploy.yml) para disparar el despliegue en Render cuando la CI termina correctamente sobre `main`.

Para activarlo en GitHub:

1. Crea en Render un `Deploy Hook` para el servicio.
2. Guarda la URL en el secret del repositorio `RENDER_DEPLOY_HOOK_URL`.
3. Protege la rama `main` y exige que la CI pase antes de hacer merge.

Sin ese secret, el workflow de deploy fallara de forma explicita para que no pase desapercibido.


## Como lanzar la web en local

1. Crea un entorno virtual:

```bash
python -m venv .venv
```

2. Activa el entorno virtual en Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

Si quieres ejecutar la misma calidad que CI en local:

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

4. Genera o actualiza los datos procesados:

```bash
python scripts/process_data.py
```

5. Lanza la aplicacion en local:

```bash
streamlit run dashboard/app.py
```

6. Abre en el navegador la URL que muestre Streamlit, normalmente:

```text
http://localhost:8501
```
