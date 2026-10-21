<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-inverse.svg">
    <img src="docs/assets/logo.svg" alt="Veyra" height="72">
  </picture>
</p>

<p align="center"><b>Worker de GPU para Veyra: entrena, predice y detecta en vivo en el equipo que tiene la tarjeta.</b></p>

<p align="center">
  <a href="LICENSE"><img alt="Licencia: AGPL-3.0" src="https://img.shields.io/badge/licencia-AGPL--3.0-78455F"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-2B2029">
  <img alt="Docker + NVIDIA" src="https://img.shields.io/badge/Docker-NVIDIA%20GPU-2B2029">
</p>

# Veyra worker

Servicio que corre en el equipo con **GPU** y ejecuta, por encargo de la API de [Veyra](https://github.com/mgueregath/veyra) (que puede estar en otro equipo):

- **entrenamientos** YOLO,
- **predicciones por lotes** (preanotar, evaluar, ejecutar sobre imágenes),
- **sesiones en vivo** sobre video, RTSP o MJPEG, con un modelo o con un **flujo** (varios modelos y procedimientos combinados con O / Y),
- **espacios de trabajo** por proyecto (VS Code por SSH) y **procedimientos** de visión que corren aislados en contenedores.

No tiene acceso a la base de datos: la API le entrega trabajos por HTTP (ver [docs/PROTOCOL.md](docs/PROTOCOL.md)) y ambos
comparten una carpeta (NFS) donde viven imágenes, videos, modelos y resultados.

**Contenido:** [Puesta en marcha](#puesta-en-marcha) · [Espacios de trabajo](#espacios-de-trabajo-vs-code-por-ssh) · [Procedimientos](#procedimientos-funciones-del-espacio-de-trabajo-dentro-de-los-flujos-de-veyra) · [Desarrollo](#desarrollo) · [Seguridad](#seguridad) · [Licencia](#licencia)

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

## Procedimientos (funciones del espacio de trabajo dentro de los flujos de Veyra)

Con los espacios activos, el worker también ejecuta **procedimientos**: funciones de visión del repositorio git de un proyecto,
declaradas en su `veyra.yaml` (`procedures: [{name, entry: módulo:función, params}]`). Cada ejecución corre en un contenedor
desechable de la imagen del espacio, sin red, con el repositorio y los datos en solo lectura, y usa el código de un **commit**
concreto (HEAD o una etiqueta), así que el resultado es reproducible. Prueba real: `python scripts/e2e_procedure.py`.

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
- Solo ejecuta tipos de trabajo conocidos (entrenar, predecir, en vivo, espacio de trabajo y procedimiento), nunca comandos arbitrarios.
- Rechaza rutas fuera de la carpeta compartida (también por enlaces simbólicos) y fuentes `file:`/`concat:`/`data:`.
- Los flujos en vivo exigen el token de la sesión. El contenedor no corre como root.

## Licencia

Este worker es software libre bajo la **GNU Affero General Public License v3.0** ([AGPL-3.0](LICENSE)), la misma licencia de
Ultralytics (YOLO), que es lo que ejecuta para entrenar y detectar. Si lo modificas y ofreces el servicio a otras personas por una
red, debes ofrecerles el código fuente de tu versión. Los logotipos de `docs/assets/` son del kit de marca de Veyra; la tipografía
DejaVu conserva su licencia ([FONT-LICENSE.txt](docs/assets/FONT-LICENSE.txt)).
