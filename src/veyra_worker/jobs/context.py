"""Lo que un manejador puede hacer con su trabajo: informar avance (con límite de frecuencia) y saber si lo cancelaron."""

import threading
import time

from ..api import ApiClient, ApiError

LOG_CAP = 60_000


class JobContext:
    def __init__(self, api: ApiClient, job_id: int, cancel: threading.Event, min_interval: float = 2.0):
        self.api, self.job_id, self.cancel = api, job_id, cancel
        self.min_interval = min_interval
        self._pending: dict = {}
        self._last = 0.0
        self.log = ""

    def add_log(self, line: str) -> None:
        self.log = (self.log + line)[-LOG_CAP:]

    def progress(self, force: bool = False, **fields) -> None:
        self._pending.update(fields)
        now = time.monotonic()
        if not self._pending or (not force and now - self._last < self.min_interval):
            return
        body = {**self._pending, "log": self.log}
        self._pending = {}
        self._last = now
        try:
            self.api.progress(self.job_id, body)
        except (ApiError, OSError):
            pass  # un fallo de red no debe matar el entrenamiento: el siguiente aviso lo recupera
        except Exception:  # httpx.TransportError y similares
            pass
