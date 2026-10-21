#!/usr/bin/env python3
"""Prueba real (con GPU y modelo de verdad) del worker contra la API simulada: predicción en lote y sesión en vivo.

    docker run --rm --device nvidia.com/gpu=all -v <carpeta_de_datos>:/data -v $PWD:/repo:ro \
        --entrypoint python veyra-worker:0.1.0 /repo/scripts/smoke_real.py <pesos.pt> <imagen>... -- <video>
Las rutas deben estar dentro de /data.
"""

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from fakeapi import FakeApi  # noqa: E402

from veyra_worker.api import ApiClient  # noqa: E402
from veyra_worker.config import Settings  # noqa: E402
from veyra_worker.runner import Worker  # noqa: E402

args = sys.argv[1:]
weights, rest = args[0], args[1:]
imgs, video = rest[: rest.index("--")], rest[rest.index("--") + 1]
s = Settings(api_url="http://api.test", token="secreto", name="smoke", data_root=Path("/data"), poll_wait=0.2,
             heartbeat_every=0.2, advertise_host="127.0.0.1", live_ports=(19200, 19260))  # fmt: skip
fake = FakeApi()
w = Worker(s, ApiClient(s.api_url, s.token, transport=httpx.MockTransport(fake)))
threading.Thread(target=w.serve, daemon=True).start()


def wait(cond, t=120):
    end = time.time() + t
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.2)
    return False


out = "/data/_smoke_pred.json"
t0 = time.time()
jid = fake.submit("predict", {"weights": weights, "images": imgs, "out": out, "imgsz": 1280, "conf": 0.05})
assert wait(lambda: jid in fake.finished), "la predicción no terminó"
f = fake.finished[jid]
print("predict:", f["status"], f["result"], f"{time.time() - t0:.1f} s")
assert f["status"] == "done", f["log"][-500:]
raw = json.load(open(out))
print("  cajas por imagen:", {Path(k).name: len(v) for k, v in raw.items()})

jid = fake.submit(
    "live",
    {
        "source": video,
        "detector": "yolo",
        "weights": weights,
        "classes": ["louse", "fly"],
        "threshold": 0.05,
        "imgsz": 640,
        "loop": True,
    },
)
assert wait(lambda: any("endpoint" in p for p in fake.progress.get(jid, []))), "la sesión no arrancó: " + str(
    fake.progress.get(jid)
)
info = next(p for p in fake.progress[jid] if "endpoint" in p)
stats_url = f"{info['endpoint']}/stats?token={info['token']}"
time.sleep(8)
st = json.load(urllib.request.urlopen(stats_url, timeout=5))
print("live:", {k: st[k] for k in ("frames", "fps", "latency_ms", "size", "source_fps")})
jpg = urllib.request.urlopen(f"{info['endpoint']}/stream.mjpg?token={info['token']}", timeout=10).read(40000)
print("  flujo MJPEG recibido:", len(jpg), "bytes, cabecera OK:", b"--frame" in jpg and b"image/jpeg" in jpg)
assert st["frames"] > 0 and b"--frame" in jpg
fake.cancel.add(jid)
assert wait(lambda: jid in fake.finished, 30)
print("  cierre:", fake.finished[jid]["status"])
Path(out).unlink(missing_ok=True)
w.shutdown(5)
print("SMOKE OK")
