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
        return s
