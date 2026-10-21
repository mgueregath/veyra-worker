"""Ejecución de un proceso hijo con salida línea a línea y cancelación del grupo completo."""

import os
import signal
import subprocess
import threading
from collections.abc import Callable


def terminate(p: subprocess.Popen, grace: float = 10.0) -> None:
    """SIGTERM al grupo del proceso y, si no cede, SIGKILL."""
    if p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_streaming(
    cmd: list[str], on_line: Callable[[str], None], cancel: threading.Event, env: dict | None = None
) -> tuple[int, bool]:
    """Devuelve (código de salida, fue_cancelado). Cada línea de stdout/stderr pasa por `on_line`."""
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True, env=env
    )
    done = threading.Event()

    def watch() -> None:
        while not done.is_set():
            if cancel.wait(0.2):
                terminate(p)
                return

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    for line in p.stdout:  # type: ignore[union-attr]
        on_line(line)
    code = p.wait()
    done.set()
    return code, cancel.is_set()
