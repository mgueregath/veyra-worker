import json
import shutil
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from veyra_worker.api import ApiClient
from veyra_worker.config import Settings
from veyra_worker.runner import Worker

HERE = Path(__file__).parent
REAL_SCRIPTS = Path(__file__).resolve().parents[1] / "src" / "veyra_worker" / "scripts"


class FakeApi:
    """La API de Veyra, simulada: una cola de trabajos y el registro de lo que el worker informa."""

    def __init__(self):
        self.queue: list[dict] = []
        self.progress: dict[int, list[dict]] = {}
        self.finished: dict[int, dict] = {}
        self.cancel: set[int] = set()
        self.registered: dict | None = None
        self.heartbeats: list[dict] = []
        self.reject_token = False
        self.lock = threading.Lock()

    def submit(self, kind: str, payload: dict) -> int:
        with self.lock:
            jid = len(self.queue) + 100
            self.queue.append({"id": jid, "kind": kind, "payload": payload})
            return jid

    def __call__(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path.removeprefix("/api/worker")
        body = json.loads(req.content or b"{}")
        if self.reject_token or req.headers.get("authorization") != "Bearer secreto":
            return httpx.Response(401, json={"detail": "token inválido"})
        with self.lock:
            if path == "/register":
                self.registered = body
                return httpx.Response(200, json={"worker_id": 7})
            if path == "/heartbeat":
                self.heartbeats.append(body)
                return httpx.Response(200, json={"cancel_jobs": sorted(self.cancel)})
            if path == "/jobs/claim":
                for j in self.queue:
                    if j["kind"] in body["kinds"] and j["id"] not in self.finished and not j.get("taken"):
                        j["taken"] = True
                        return httpx.Response(200, json=j)
                return httpx.Response(204)
            jid = int(path.split("/")[2])
            if path.endswith("/progress"):
                self.progress.setdefault(jid, []).append(body)
                return httpx.Response(200, json={"cancel": jid in self.cancel})
            if path.endswith("/finish"):
                self.finished[jid] = body
                return httpx.Response(200, json={"ok": True})
        return httpx.Response(404, json={"detail": path})


@pytest.fixture()
def env(tmp_path):
    """Worker real contra la API simulada, con scripts simulados para entrenar y predecir (sin GPU)."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(HERE / "fake_train.py", scripts / "train_yolo.py")
    shutil.copy(HERE / "fake_predict.py", scripts / "predict_yolo.py")
    shutil.copy(REAL_SCRIPTS / "live_yolo.py", scripts / "live_yolo.py")
    shutil.copy(REAL_SCRIPTS / "live_core.py", scripts / "live_core.py")
    data = tmp_path / "data"
    data.mkdir()
    s = Settings(api_url="http://api.test", token="secreto", name="test", data_root=data, python=sys.executable,
                 scripts_dir=scripts, advertise_host="127.0.0.1", live_ports=(19100, 19160), poll_wait=0.2,
                 heartbeat_every=0.1, live_start_timeout=30)  # fmt: skip
    fake = FakeApi()
    api = ApiClient(s.api_url, s.token, transport=httpx.MockTransport(fake))
    w = Worker(s, api)
    t = threading.Thread(target=w.serve, daemon=True)
    t.start()

    def wait(cond, timeout=20.0):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return True
            time.sleep(0.05)
        return False

    yield type("Env", (), {"s": s, "fake": fake, "w": w, "wait": staticmethod(wait), "data": data})
    w.shutdown(5)
    t.join(5)
    api.close()
