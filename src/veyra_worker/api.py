"""Cliente HTTP del protocolo worker↔API (ver docs/PROTOCOL.md)."""

import httpx

from . import __version__


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status


class ApiClient:
    def __init__(self, base_url: str, token: str, transport: httpx.BaseTransport | None = None, timeout: float = 30.0):
        self.http = httpx.Client(
            base_url=base_url + "/api/worker",
            headers={"Authorization": f"Bearer {token}", "User-Agent": f"veyra-worker/{__version__}"},
            timeout=timeout,
            transport=transport,
        )

    def _call(self, method: str, path: str, body: dict | None = None, timeout: float | None = None):
        r = self.http.request(method, path, json=body, timeout=timeout)
        if r.status_code == 204:
            return None
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(r.status_code, str(detail))
        return r.json()

    def register(self, info: dict) -> dict:
        return self._call("POST", "/register", info)

    def heartbeat(self, running: list[int], busy: dict) -> dict:
        return self._call("POST", "/heartbeat", {"running_jobs": running, **busy})

    def claim(self, kinds: list[str], wait: float) -> dict | None:
        """Pide un trabajo; la API puede retener la petición hasta `wait` segundos. None = no hay nada."""
        return self._call("POST", "/jobs/claim", {"kinds": kinds, "wait": wait}, timeout=wait + 15)

    def progress(self, job_id: int, body: dict) -> dict:
        return self._call("POST", f"/jobs/{job_id}/progress", body)

    def finish(self, job_id: int, status: str, result: dict | None = None, error: str = "", log: str = "") -> None:
        self._call("POST", f"/jobs/{job_id}/finish", {"status": status, "result": result, "error": error, "log": log})

    def close(self) -> None:
        self.http.close()
