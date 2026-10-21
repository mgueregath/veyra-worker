"""Configuración del worker: todo viene de variables de entorno (VEYRA_*), pensado para un contenedor."""

import os
import socket
import sys
from dataclasses import dataclass, field
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"


def _port_range(text: str) -> tuple[int, int]:
    a, _, b = text.partition("-")
    lo, hi = int(a), int(b or a)
    if not 1024 <= lo <= hi <= 65535:
        raise ValueError(f"VEYRA_LIVE_PORTS inválido: {text!r}")
    return lo, hi


@dataclass
class Settings:
    api_url: str
    token: str
    name: str = field(default_factory=socket.gethostname)
    data_root: Path = Path("/data")  # carpeta compartida con la API (NFS): TODAS las rutas de los trabajos viven aquí
    device: str = "0"
    python: str = sys.executable  # intérprete con PyTorch/Ultralytics que ejecuta los scripts
    scripts_dir: Path = SCRIPTS
    advertise_host: str = ""  # dirección con la que la API alcanza a este equipo (para los flujos en vivo)
    live_ports: tuple[int, int] = (9000, 9099)
    max_live: int = 2  # sesiones en vivo simultáneas
    max_batch: int = 1  # entrenamientos/predicciones a la vez (una GPU)
    poll_wait: float = 20.0  # segundos que la API retiene una petición de trabajo
    heartbeat_every: float = 5.0
    live_start_timeout: float = 120.0
    pretrained_dirs: tuple[str, ...] = ()  # carpetas con pesos preentrenados (además de la carpeta compartida)
    # espacios de trabajo (VS Code por SSH): contenedores que este worker lanza con el Docker del equipo
    workspaces: bool = False
    workspace_image: str = "docker.labs.hapticus.cl/veyra-workspace:0.3.0"
    workspace_ports: tuple[int, int] = (9100, 9199)
    data_volume: str = ""  # volumen de Docker con la carpeta compartida (se monta en /data, solo lectura)
    max_workspaces: int = 4
    max_procedures: int = 2
    docker_bin: str = "docker"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env
        missing = [k for k in ("VEYRA_API_URL", "VEYRA_WORKER_TOKEN") if not e.get(k)]
        if missing:
            raise SystemExit("Faltan variables de entorno: " + ", ".join(missing))
        s = cls(api_url=e["VEYRA_API_URL"].rstrip("/"), token=e["VEYRA_WORKER_TOKEN"])
        if e.get("VEYRA_WORKER_NAME"):
            s.name = e["VEYRA_WORKER_NAME"]
        s.data_root = Path(e.get("VEYRA_DATA_ROOT", str(s.data_root)))
        s.device = e.get("VEYRA_DEVICE", s.device)
        s.python = e.get("VEYRA_PYTHON", s.python)
        s.advertise_host = e.get("VEYRA_ADVERTISE_HOST", "")
        s.live_ports = _port_range(e.get("VEYRA_LIVE_PORTS", "9000-9099"))
        s.heartbeat_every = float(e.get("VEYRA_HEARTBEAT", s.heartbeat_every))
        s.max_live = int(e.get("VEYRA_MAX_LIVE", s.max_live))
        s.max_batch = int(e.get("VEYRA_MAX_BATCH", s.max_batch))
        s.pretrained_dirs = tuple(x for x in e.get("VEYRA_PRETRAINED_DIRS", "").split(":") if x)
        s.workspaces = e.get("VEYRA_WORKSPACES", "0") in ("1", "true", "yes")
        s.workspace_image = e.get("VEYRA_WORKSPACE_IMAGE", s.workspace_image)
        s.workspace_ports = _port_range(e.get("VEYRA_WORKSPACE_PORTS", "9100-9199"))
        s.data_volume = e.get("VEYRA_DATA_VOLUME", "")
        s.max_workspaces = int(e.get("VEYRA_MAX_WORKSPACES", s.max_workspaces))
        s.max_procedures = int(e.get("VEYRA_MAX_PROCEDURES", s.max_procedures))
        s.docker_bin = e.get("VEYRA_DOCKER_BIN", s.docker_bin)
        return s
