# Veyra worker

Servicio que corre en el equipo con **GPU** y ejecuta, por encargo de la API de Veyra (que puede estar en otro equipo):

- **entrenamientos** YOLO,
- **predicciones por lotes** (preanotar, evaluar, ejecutar sobre imágenes),
- **sesiones en vivo** sobre video, RTSP o MJPEG.

No tiene acceso a la base de datos: la API le entrega trabajos por HTTP (ver [docs/PROTOCOL.md](docs/PROTOCOL.md)) y ambos
comparten una carpeta (NFS) donde viven imágenes, videos, modelos y resultados.

## Puesta en marcha

```bash
cp .env.example .env            # completar VEYRA_API_URL y VEYRA_WORKER_TOKEN
docker compose run --rm worker check    # diagnóstico: carpeta compartida, GPU, versiones, API y token
docker compose pull && docker compose up -d     # imagen docker.labs.hapticus.cl/veyra-worker (o `--build` para construirla aquí)
```

Requisitos del equipo: Docker con acceso a la GPU por CDI (`docker run --device nvidia.com/gpu=all …`) y que el
export NFS exista en el NAS: por defecto `:/mnt/main/veyra` (hoy el NAS solo exporta `heimdall` y `vault`; ver `.env.example`).

## Espacios de trabajo (VS Code por SSH)

Con `VEYRA_WORKSPACES=1` el worker también ofrece **espacios de trabajo**: un contenedor por proyecto de Veyra (`veyra-workspace`)
al que se entra con VS Code Remote-SSH. Se enciende y apaga desde Veyra (Gestión → Espacio de trabajo), se apaga solo tras un
rato sin conexiones ni carga, y todo lo de `/workspace` se conserva en un volumen propio del proyecto:

- `/workspace/project`: repositorio git del proyecto (al apagar se guarda una copia en `refs/veyra/autosave/*`).
- `/data` (solo lectura): imágenes, videos y modelos de Veyra en el NFS. Entorno virtual persistente: `pip install …`; paquetes del sistema: `veyra-apt install …`.
- Acceso solo con las llaves SSH públicas de quienes administran el proyecto (se registran en Veyra); sin contraseña.

Requiere el socket de Docker del equipo (ver `docker-compose.yml`) y construir la imagen: `docker build -t docker.labs.hapticus.cl/veyra-workspace:<versión> --build-arg BASE=docker.labs.hapticus.cl/veyra-worker:<versión> workspace`.
Prueba real con Docker: `python scripts/e2e_workspace.py`.

## Desarrollo

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
pytest            # el worker real contra una API simulada, con scripts simulados (sin GPU)
ruff check .
```

Los scripts de GPU (`src/veyra_worker/scripts/`) tienen la misma interfaz de línea de comandos que usaba la API al lanzarlos
en local (`PROGRESS {…}` / `RESULT {…}` por la salida estándar). `live_yolo.py` gana `--host` para escuchar en la red.

## Seguridad

- El token del worker se guarda como hash en la API; se envía por la red privada (ZeroTier/LAN).
- Solo ejecuta tres tipos de trabajo conocidos, nunca comandos arbitrarios.
- Rechaza rutas fuera de la carpeta compartida (también por enlaces simbólicos) y fuentes `file:`/`concat:`/`data:`.
- Los flujos en vivo exigen el token de la sesión. El contenedor no corre como root.
- Ultralytics es AGPL-3.0: ver el aviso en `train_yolo.py`.
