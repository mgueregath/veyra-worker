"""Sesión en vivo: lanza scripts/live_yolo.py escuchando en la red con un token y le cuenta a la API dónde está.

El trabajo queda «en curso» mientras la sesión viva; la API reenvía el flujo MJPEG al navegador y se entiende
directamente con el proceso (estadísticas y controles). Cancelar el trabajo detiene el proceso.
"""

import json
import os
import re
import secrets
import socket
import subprocess
import threading
import time

from .. import paths
from ..config import Settings
from ..proc import terminate
from . import Outcome
from .context import JobContext
from .procedure import NAME, REF
from .workspace import SLUG, WorkspaceError

_port_lock = threading.Lock()
_ports_in_use: set[int] = set()


def _pick_port(lo: int, hi: int) -> int:
    with _port_lock:
        for p in range(lo, hi + 1):
            if p in _ports_in_use:
                continue
            with socket.socket() as k:
                try:
                    k.bind(("0.0.0.0", p))
                except OSError:
                    continue
            _ports_in_use.add(p)
            return p
    raise RuntimeError(f"no quedan puertos libres en {lo}-{hi}")


def _primary_ip() -> str:
    """La dirección con la que este equipo sale a la red (sin enviar nada)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as k:
        try:
            k.connect(("10.255.255.255", 1))
            return k.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def build_cmd(job: dict, s: Settings, port: int, token: str) -> list[str]:
    a = job["payload"]
    root = s.data_root
    detector = a.get("detector", "yolo" if a.get("weights") else "none")
    if detector not in ("yolo", "fake", "none"):
        raise ValueError(f"detector desconocido: {detector}")
    cmd = [
        s.python, str(s.scripts_dir / "live_yolo.py"),
        "--source", paths.source(root, a["source"]), "--detector", detector,
        "--host", "0.0.0.0", "--port", str(port), "--token", token,
        "--threshold", str(float(a.get("threshold", 0.25))), "--imgsz", str(int(a.get("imgsz", 1280))),
        "--device", s.device, "--classes", json.dumps(a.get("classes", [])),
    ]  # fmt: skip
    if a.get("record_dir"):
        cmd += ["--record-dir", str(paths.inside(root, a["record_dir"]))]
    if a.get("weights") and detector == "yolo":
        cmd += ["--weights", paths.weights(root, s.pretrained_dirs, a["weights"])]
    if a.get("loop", True) and "://" not in a["source"]:
        cmd += ["--loop"]
    if a.get("no_realtime"):
        cmd += ["--no-realtime"]
    return cmd


def _flow_node(node: dict, models: dict, depth: int = 0) -> dict:
    """Copia validada de un nodo del flujo: solo campos conocidos, fuentes admitidas y nombres bien formados."""
    if depth > 8 or not isinstance(node, dict):
        raise WorkspaceError("flujo inválido")
    if "op" in node:
        if node["op"] not in ("or", "and") or not 1 <= len(node.get("inputs", [])) <= 8:
            raise WorkspaceError("flujo inválido: operación o entradas")
        out = {"op": node["op"], "inputs": [_flow_node(c, models, depth + 1) for c in node["inputs"]]}
        for k in ("iou",):
            if node.get(k) is not None:
                out[k] = float(node[k])
        if node.get("score") is not None:
            if node["score"] not in ("max", "min", "mean"):
                raise WorkspaceError("flujo inválido: puntaje")
            out["score"] = node["score"]
        return out
    src = str(node.get("source", ""))
    leaf: dict = {"source": src, "min_score": float(node.get("min_score") or 0.0)}
    if src.startswith("model:"):
        if src[6:] not in models:
            raise WorkspaceError(f"el flujo usa un modelo sin pesos: {src}")
    elif src.startswith("procedure:"):
        if not NAME.match(src[10:]):
            raise WorkspaceError("nombre de procedimiento inválido")
        ref = str(node.get("ref") or "HEAD")
        if not REF.match(ref):
            raise WorkspaceError("versión de procedimiento inválida")
        leaf["ref"] = ref
        params = node.get("params") or {}
        if not isinstance(params, dict):
            raise WorkspaceError("los parámetros deben ser un objeto")
        leaf["params"] = params
    else:
        raise WorkspaceError(f"fuente no admitida en vivo: {src}")
    return leaf


def _in_data(s: Settings, p: str) -> str:
    """Ruta del worker -> ruta dentro del contenedor (la carpeta compartida se monta en /data)."""
    root = s.data_root.resolve()
    return f"/data/{paths.inside(root, p).relative_to(root)}"


def build_flow_cmd(job: dict, s: Settings, port: int, token: str, name: str) -> list[str]:
    """Sesión con flujo: corre en un contenedor de la imagen del espacio (con el entorno del proyecto), repositorio y datos
    en solo lectura. Con red porque las cámaras la necesitan; solo se publica el puerto de la sesión."""
    a = job["payload"]
    flow = a["flow"]
    slug = str(flow.get("slug", ""))
    if not SLUG.match(slug):
        raise WorkspaceError(f"proyecto inválido: {slug!r}")
    if not s.data_volume:
        raise WorkspaceError("este worker no tiene configurado VEYRA_DATA_VOLUME (necesario para flujos en vivo)")
    models = {}
    for mid, info in (flow.get("models") or {}).items():
        if not re.fullmatch(r"\d{1,9}", str(mid)):
            raise WorkspaceError("modelo inválido en el flujo")
        models[str(mid)] = {"path": _in_data(s, info["path"]), "classes": [str(c) for c in info.get("classes", [])]}
    bundle = {"spec": _flow_node(flow["spec"], models), "models": models}
    src = a["source"]
    source = src if "://" in src else _in_data(s, src)
    if "://" in src:
        paths.source(s.data_root, src)  # solo esquemas permitidos
    cmd = [
        s.docker_bin, "run", "--rm", "--init", "--name", name, "--user", "1000:3000",
        "--cpus", str(a.get("cpus", 4)), "--memory", str(a.get("memory", "8g")), "--pids-limit", "1024",
        "-p", f"{port}:{port}", "-v", f"veyra-ws-{slug}:/workspace:ro", "-v", f"{s.data_volume}:/data:ro",
        "--device", "nvidia.com/gpu=all", "-e", "VEYRA_FLOW=" + json.dumps(bundle),
    ]  # fmt: skip
    live = [
        "--source", source, "--flow", "--host", "0.0.0.0", "--port", str(port), "--token", token,
        "--imgsz", str(int(a.get("imgsz", 1280))), "--device", "0",
    ]  # fmt: skip
    if a.get("record_dir"):
        rec = paths.inside(s.data_root, a["record_dir"])
        rel = rec.relative_to(s.data_root.resolve())
        rec.mkdir(parents=True, exist_ok=True)
        cmd += ["--mount", f"type=volume,src={s.data_volume},dst=/rec,volume-subpath={rel}"]
        live += ["--record-dir", "/rec"]
    if a.get("loop", True) and "://" not in src:
        live += ["--loop"]
    if a.get("no_realtime"):
        live += ["--no-realtime"]
    py = 'PY=/workspace/.venv/bin/python; [ -x "$PY" ] || PY=python; exec "$PY" -m veyra_worker.scripts.live_yolo "$@"'
    return cmd + ["--entrypoint", "sh", s.workspace_image, "-c", py, "sh", *live]


def run(ctx: JobContext, job: dict, s: Settings) -> Outcome:
    port, token = _pick_port(*s.live_ports), secrets.token_hex(24)
    container = ""
    try:
        if job["payload"].get("flow"):
            container = f"veyra-live-{token[:12]}"
            cmd = build_flow_cmd(job, s, port, token, container)
        else:
            cmd = build_cmd(job, s, port, token)
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True, env={**os.environ}
        )
        ready = threading.Event()

        def pump() -> None:
            for line in p.stdout:  # type: ignore[union-attr]
                ctx.add_log(line)
                if line.startswith("LISTO"):
                    ready.set()

        threading.Thread(target=pump, daemon=True).start()
        deadline = time.time() + s.live_start_timeout
        while not ready.is_set() and p.poll() is None and time.time() < deadline and not ctx.cancel.is_set():
            ready.wait(0.2)
        if ctx.cancel.is_set():
            terminate(p)
            return Outcome("cancelled", log=ctx.log)
        if not ready.is_set():
            terminate(p)
            return Outcome("failed", error="el proceso de detección no arrancó: " + ctx.log[-400:], log=ctx.log)
        host = s.advertise_host or _primary_ip()
        ctx.progress(force=True, endpoint=f"http://{host}:{port}", token=token, pid=p.pid)
        while p.poll() is None and not ctx.cancel.is_set():
            time.sleep(0.3)
        if ctx.cancel.is_set():
            terminate(p)
            return Outcome("cancelled", log=ctx.log)
        if p.returncode == 0:
            return Outcome("done", result={"ended": True}, log=ctx.log)
        return Outcome("failed", error=f"el proceso terminó con código {p.returncode}", log=ctx.log)
    except (WorkspaceError, paths.PathError, KeyError, ValueError) as e:
        return Outcome("failed", error=str(e), log=ctx.log)
    finally:
        if container:
            subprocess.run([s.docker_bin, "rm", "-f", container], capture_output=True)
        with _port_lock:
            _ports_in_use.discard(port)
