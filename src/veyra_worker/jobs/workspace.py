"""Espacio de trabajo de un proyecto: un contenedor con SSH para abrirlo desde VS Code (Remote-SSH).

El trabajo queda «en curso» mientras el espacio esté encendido. Se apaga solo tras `idle_minutes` sin conexiones SSH ni
carga de CPU, o al cancelar el trabajo. Todo lo que hay en /workspace (volumen propio del proyecto) se conserva, y antes de
apagar se guarda una copia automática del código en git (refs/veyra/autosave/*).

Los parámetros que llegan de la API NO se pasan tal cual a Docker: se validan y el comando lo arma este módulo.
"""

import re
import socket
import subprocess
import time

from ..config import Settings
from . import Outcome
from .context import JobContext

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
KEY = re.compile(
    r"^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(?:256|384|521)|sk-ssh-ed25519@openssh\.com) [A-Za-z0-9+/=]+(?: [^\r\n]*)?$"
)
MEMORY = re.compile(r"^[1-9][0-9]{0,3}[mg]$")
CPU_BUSY = 30.0  # % de CPU del contenedor a partir del cual se considera que hay trabajo en marcha


class WorkspaceError(ValueError):
    pass


def container_name(slug: str) -> str:
    return f"veyra-ws-{slug}"


def validate(payload: dict, s: Settings) -> dict:
    slug = str(payload.get("slug", ""))
    if not SLUG.match(slug):
        raise WorkspaceError(f"nombre de proyecto inválido para un espacio de trabajo: {slug!r}")
    port = int(payload.get("port", 0))
    if not s.workspace_ports[0] <= port <= s.workspace_ports[1]:
        raise WorkspaceError(f"puerto fuera del rango permitido {s.workspace_ports[0]}-{s.workspace_ports[1]}")
    keys = [k.strip() for k in payload.get("keys", []) if k and k.strip()]
    if not keys or len(keys) > 12 or not all(KEY.match(k) for k in keys):
        raise WorkspaceError("hacen falta entre 1 y 12 llaves públicas SSH válidas")
    mem = str(payload.get("memory", "12g"))
    if not MEMORY.match(mem):
        raise WorkspaceError(f"memoria inválida: {mem!r}")
    cpus = float(payload.get("cpus", 6))
    if not 0.5 <= cpus <= 64:
        raise WorkspaceError("cpus fuera de rango")
    return {"slug": slug, "port": port, "keys": keys, "memory": mem, "cpus": cpus, "gpu": bool(payload.get("gpu", True)),
            "idle": float(payload.get("idle_minutes", 45))}  # fmt: skip


def run_args(v: dict, s: Settings, host: str) -> list[str]:
    name = container_name(v["slug"])
    cmd = [
        s.docker_bin, "run", "-d", "--name", name, "--hostname", f"ws-{v['slug']}",
        "--label", f"veyra.workspace={v['slug']}", "--cpus", str(v["cpus"]), "--memory", v["memory"], "--pids-limit", "4096",
        "-p", f"{host}:{v['port']}:22", "-e", "VEYRA_AUTHORIZED_KEYS=" + "\n".join(v["keys"]), "-e", f"VEYRA_PROJECT={v['slug']}",
        "-v", f"veyra-ws-{v['slug']}:/workspace",
    ]  # fmt: skip
    if s.data_volume:
        cmd += ["-v", f"{s.data_volume}:/data:ro"]
    if v["gpu"]:
        cmd += ["--device", "nvidia.com/gpu=all"]
    return cmd + [s.workspace_image]


def _docker(s: Settings, *args: str, timeout: float = 60, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run([s.docker_bin, *args], capture_output=True, text=True, timeout=timeout, check=check)


def available(s: Settings) -> bool:
    """¿Se puede hablar con Docker? (si no, este worker no ofrece espacios de trabajo)"""
    try:
        return _docker(s, "version", "--format", "{{.Server.Version}}", timeout=8).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _running(s: Settings, name: str) -> bool:
    r = _docker(s, "container", "inspect", "-f", "{{.State.Running}}", name)
    return r.returncode == 0 and r.stdout.strip() == "true"


def _exists(s: Settings, name: str) -> bool:
    return _docker(s, "container", "inspect", name).returncode == 0


def _ready(s: Settings, name: str) -> bool:
    r = _docker(s, "logs", name)
    return "LISTO veyra-workspace" in (r.stdout + r.stderr)


def _ssh_connections(s: Settings, name: str) -> int:
    r = _docker(s, "exec", name, "sh", "-c", "ss -Htn state established '( sport = :22 )' | wc -l", timeout=15)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0


def _cpu(s: Settings, name: str) -> float:
    r = _docker(s, "stats", "--no-stream", "--format", "{{.CPUPerc}}", name, timeout=20)
    try:
        return float(r.stdout.strip().rstrip("%") or 0)
    except ValueError:
        return 0.0


def _stop(s: Settings, name: str) -> str:
    """Copia automática del código y apagado ordenado; el volumen del proyecto se conserva."""
    saved = ""
    try:
        r = _docker(s, "exec", "-u", "veyra", name, "veyra-autosave", timeout=60)
        saved = (r.stdout or r.stderr).strip()[-120:]
    except (OSError, subprocess.SubprocessError):
        pass
    _docker(s, "stop", "-t", "20", name, timeout=60)
    _docker(s, "rm", "-f", name, timeout=30)
    return saved


def _primary_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as k:
        try:
            k.connect(("10.255.255.255", 1))
            return k.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def run(ctx: JobContext, job: dict, s: Settings) -> Outcome:
    try:
        v = validate(job["payload"], s)
    except (WorkspaceError, ValueError, TypeError) as e:
        return Outcome("failed", error=str(e))
    name, host = container_name(v["slug"]), s.advertise_host or _primary_ip()
    adopted = _running(s, name)
    if adopted:  # el worker se reinició pero el espacio seguía encendido: se retoma sin cortar sesiones
        ctx.add_log("espacio ya encendido: se retoma\n")
        keys = "\n".join(v["keys"])
        _docker(
            s, "exec", "-i", name, "sh", "-c", "cat > /etc/ssh/authorized_keys_veyra", timeout=20
        ) if False else None
        subprocess.run([s.docker_bin, "exec", "-i", "-u", "root", name, "sh", "-c", "cat > /etc/ssh/authorized_keys_veyra"],
                       input=keys + "\n", text=True, capture_output=True, timeout=20)  # fmt: skip
    else:
        if _exists(s, name):
            _docker(s, "rm", "-f", name)
        _docker(s, "volume", "create", "--label", f"veyra.workspace={v['slug']}", f"veyra-ws-{v['slug']}")
        r = _docker(s, *run_args(v, s, host)[1:], timeout=120)
        if r.returncode != 0:
            return Outcome("failed", error="no se pudo crear el contenedor: " + (r.stderr or r.stdout)[-400:])
        deadline = time.time() + 150
        while time.time() < deadline and not _ready(s, name):
            if not _running(s, name):
                logs = _docker(s, "logs", "--tail", "30", name)
                _docker(s, "rm", "-f", name)
                return Outcome("failed", error="el espacio no arrancó: " + (logs.stdout + logs.stderr)[-400:])
            if ctx.cancel.wait(1.0):
                _stop(s, name)
                return Outcome("cancelled")
    ctx.progress(
        force=True,
        host=host,
        port=v["port"],
        user="veyra",
        workdir="/workspace/project",
        container=name,
        adopted=adopted,
    )
    idle_since = None
    while True:
        if ctx.cancel.wait(20.0 if v["idle"] * 60 >= 120 else 3.0):
            saved = _stop(s, name)
            return Outcome("cancelled", result={"reason": "stopped", "autosave": saved}, log=ctx.log)
        if not _running(s, name):
            _docker(s, "rm", "-f", name)
            return Outcome("failed", error="el contenedor del espacio terminó inesperadamente", log=ctx.log)
        busy = _ssh_connections(s, name) > 0 or _cpu(s, name) >= CPU_BUSY
        if busy or v["idle"] <= 0:
            idle_since = None
            continue
        idle_since = idle_since or time.time()
        if time.time() - idle_since >= v["idle"] * 60:
            saved = _stop(s, name)
            return Outcome("done", result={"reason": "idle", "autosave": saved}, log=ctx.log)
