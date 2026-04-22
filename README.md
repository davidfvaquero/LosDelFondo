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

## CD hacia AWS EC2 con GitHub Actions

Si quieres desplegar en una instancia EC2 en vez de Render, el repo incluye ahora:

- el workflow [`.github/workflows/ec2-deploy.yml`](/home/gordolinus/projects/LosDelFondo/.github/workflows/ec2-deploy.yml)
- el script remoto [`scripts/deploy_ec2.sh`](/home/gordolinus/projects/LosDelFondo/scripts/deploy_ec2.sh)

El flujo es:

1. Abres o actualizas una PR contra `main`, `dev` o `api`.
2. GitHub ejecuta la CI sobre la rama origen de esa PR.
3. Si la CI termina bien, dentro del mismo workflow se ejecuta el despliegue por SSH hacia la EC2 del entorno asociado a la rama destino.
4. La EC2 hace `git fetch`, actualiza la rama destino de la PR, instala dependencias, regenera datos y reinicia el servicio.

### Environments y ramas

Configura tres `Environments` en GitHub:

- `Production` para la rama `main`
- `staging` para la rama `dev`
- `api` para la rama `api`

El job de deploy selecciona automaticamente el environment segun la rama destino de la PR:

- `main` -> `Production`
- `dev` -> `staging`
- `api` -> `api`

Si vas a desplegar desde PRs, no limites `Deployment branches and tags` solo a `main`, `dev` o `api`, porque el workflow se ejecuta tomando como referencia la rama origen de la PR.

Opciones recomendadas:

- dejar el environment sin restriccion de ramas
- o permitir un patron que cubra las ramas desde las que abras PRs

Ejemplos:

- PR `feature-x` -> `dev`: despliega `dev` en la EC2 del environment `staging`
- PR `fix-y` -> `main`: despliega `main` en la EC2 del environment `Production`
- PR `feature-api` -> `api`: despliega `api` en la EC2 del environment `api`

El workflow [`.github/workflows/ec2-deploy.yml`](/home/gordolinus/projects/LosDelFondo/.github/workflows/ec2-deploy.yml) queda como utilidad manual para lanzarlo desde `workflow_dispatch` si necesitas forzar un deploy de una rama concreta.

### Secrets que debes crear en GitHub

En `Settings > Environments`, entra en cada environment y crea estos mismos secrets, pero con los valores de la EC2 correspondiente:

- `EC2_HOST`: IP publica o DNS de la instancia.
- `EC2_USER`: usuario SSH, por ejemplo `ubuntu`.
- `EC2_SSH_PRIVATE_KEY`: clave privada que GitHub Actions usara para entrar por SSH.
- `EC2_KNOWN_HOSTS`: salida de `ssh-keyscan -H <tu-host>`.
- `EC2_APP_DIR`: ruta absoluta donde esta clonado el repo en la EC2, por ejemplo `/opt/losdelfondo`.
- `EC2_SYSTEMD_SERVICE`: nombre del servicio `systemd` que arranca Streamlit, por ejemplo `losdelfondo`.

Ejemplo:

- environment `Production`: secretos de la EC2 de produccion
- environment `staging`: secretos de la EC2 de pruebas
- environment `api`: secretos de la EC2 donde vive la API

### Preparacion unica en la EC2

1. Clona el repo en una ruta fija, por ejemplo `/opt/losdelfondo`.
2. Asegurate de que el usuario de `EC2_USER` puede ejecutar `sudo systemctl restart <servicio>` sin pedir password.
3. Crea el servicio `systemd`.

Ejemplo de unidad `systemd` en `/etc/systemd/system/losdelfondo.service`:

```ini
[Unit]
Description=LosDelFondo Streamlit app
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/losdelfondo
Environment="PATH=/opt/losdelfondo/.venv/bin"
ExecStart=/opt/losdelfondo/.venv/bin/streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Luego en la EC2:

```bash
sudo systemctl daemon-reload
sudo systemctl enable losdelfondo
sudo systemctl start losdelfondo
```

### Comandos utiles

Para generar `EC2_KNOWN_HOSTS` desde tu maquina local:

```bash
ssh-keyscan -H TU_HOST_EC2
```

Para permitir reinicio sin password al usuario `ubuntu`, abre `sudo visudo` y anade algo como:

```text
ubuntu ALL=NOPASSWD:/bin/systemctl restart losdelfondo,/bin/systemctl status losdelfondo
```

Si prefieres otro servicio o usuario, ajusta el nombre tanto en la instancia como en el secret `EC2_SYSTEMD_SERVICE`.


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
