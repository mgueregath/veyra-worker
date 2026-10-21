"""Línea de comandos: `veyra-worker run` (servicio), `veyra-worker check` (diagnóstico) y `version`."""

import argparse
import logging
import signal
import sys
import threading

from . import __version__
from .api import ApiClient, ApiError
from .config import Settings
from .runner import Worker
from .system import capabilities, gpus


def check(s: Settings) -> int:
    ok = True

    def line(name: str, good: bool, extra: str = "") -> None:
        nonlocal ok
        ok &= good
        print(("OK     " if good else "FALLA  ") + name + (f"  {extra}" if extra else ""))

    line("carpeta compartida existe", s.data_root.is_dir(), str(s.data_root))
    if s.data_root.is_dir():
        probe = s.data_root / f".veyra-worker-{s.name}.probe"
        try:
            probe.write_text("ok")
            probe.unlink()
            line("se puede escribir en la carpeta compartida", True)
        except OSError as e:
            line("se puede escribir en la carpeta compartida", False, str(e))
    g = gpus()
    line("GPU visible", bool(g), ", ".join(f"{x['name']} {x['memory_mb']} MB" for x in g) or "nvidia-smi no ve ninguna")
    caps = capabilities(s)
    line("PyTorch", bool(caps["torch"]), str(caps["torch"]))
    line("Ultralytics", bool(caps["ultralytics"]), str(caps["ultralytics"]))
    for f in ("train_yolo.py", "predict_yolo.py", "live_yolo.py"):
        line(f"script {f}", (s.scripts_dir / f).is_file())
    api = ApiClient(s.api_url, s.token)
    try:
        r = api.register(
            {"name": s.name, "kinds": [], "max_batch": 0, "max_live": 0, "capabilities": caps, "check_only": True}
        )
        line("la API responde y acepta el token", True, f"worker {r.get('worker_id')}")
    except ApiError as e:
        line("la API responde y acepta el token", False, str(e))
    except Exception as e:
        line("la API responde y acepta el token", False, f"{type(e).__name__}: {e}")
    finally:
        api.close()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="veyra-worker")
    ap.add_argument("command", choices=["run", "check", "version"])
    a = ap.parse_args(argv)
    if a.command == "version":
        print(__version__)
        return 0
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = Settings.from_env()
    if a.command == "check":
        return check(s)
    w = Worker(s, ApiClient(s.api_url, s.token))
    done = threading.Event()

    def bye(*_):
        done.set()
        w.shutdown()

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    logging.getLogger("veyra_worker").info(
        "Veyra worker %s · API %s · datos en %s", __version__, s.api_url, s.data_root
    )
    w.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
