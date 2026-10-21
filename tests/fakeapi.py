"""La API de Veyra simulada (la usan los tests y scripts/smoke_real.py)."""

import json
import threading

import httpx


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
