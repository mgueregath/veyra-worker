"""Sesión en vivo: lanza scripts/live_yolo.py escuchando en la red con un token y le cuenta a la API dónde está.

El trabajo queda «en curso» mientras la sesión viva; la API reenvía el flujo MJPEG al navegador y se entiende
directamente con el proceso (estadísticas y controles). Cancelar el trabajo detiene el proceso.
"""

import json
import os
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


def run(ctx: JobContext, job: dict, s: Settings) -> Outcome:
    port, token = _pick_port(*s.live_ports), secrets.token_hex(24)
    try:
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
    finally:
        with _port_lock:
            _ports_in_use.discard(port)
