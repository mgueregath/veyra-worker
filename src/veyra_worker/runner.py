"""Bucle principal: registrarse, avisar que sigue vivo, pedir trabajos y ejecutarlos (cada uno en su hilo)."""

import logging
import threading
import time
from dataclasses import dataclass

from .api import ApiClient, ApiError
from .config import Settings
from .jobs import Outcome, live, predict, train
from .jobs.context import JobContext
from .system import capabilities

log = logging.getLogger("veyra_worker")

HANDLERS = {"train": train.run, "predict": predict.run, "live": live.run}
BATCH_KINDS = ("train", "predict")


@dataclass
class Running:
    kind: str
    cancel: threading.Event
    thread: threading.Thread


class Worker:
    def __init__(self, settings: Settings, api: ApiClient, handlers: dict | None = None):
        self.s, self.api = settings, api
        self.handlers = handlers or HANDLERS
        self.running: dict[int, Running] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.worker_id: int | None = None

    # ---------- capacidad ----------
    def _count(self, live_kind: bool) -> int:
        with self.lock:
            return sum(1 for r in self.running.values() if (r.kind == "live") == live_kind and r.thread.is_alive())

    def free_kinds(self) -> list[str]:
        kinds = []
        if self._count(False) < self.s.max_batch:
            kinds += [k for k in BATCH_KINDS if k in self.handlers]
        if "live" in self.handlers and self._count(True) < self.s.max_live:
            kinds.append("live")
        return kinds

    # ---------- ciclo de vida ----------
    def register(self) -> None:
        info = {"name": self.s.name, "kinds": sorted(self.handlers), "max_batch": self.s.max_batch,
                "max_live": self.s.max_live, "capabilities": capabilities(self.s)}  # fmt: skip
        r = self.api.register(info)
        self.worker_id = r.get("worker_id")
        log.info("registrado como «%s» (id %s)", self.s.name, self.worker_id)

    def heartbeat_once(self) -> None:
        with self.lock:
            ids = [i for i, r in self.running.items() if r.thread.is_alive()]
        r = self.api.heartbeat(
            ids, {"free_batch": self.s.max_batch - self._count(False), "free_live": self.s.max_live - self._count(True)}
        )
        for jid in (r or {}).get("cancel_jobs", []):
            with self.lock:
                run = self.running.get(jid)
            if run:
                log.info("cancelando el trabajo %s", jid)
                run.cancel.set()

    def _heartbeat_loop(self) -> None:
        while not self.stop.wait(self.s.heartbeat_every):
            try:
                self.heartbeat_once()
            except Exception as e:  # la API puede estar reiniciándose: se reintenta en el próximo ciclo
                log.warning("latido fallido: %s", e)

    def _run_job(self, job: dict, run: Running) -> None:
        jid = job["id"]
        ctx = JobContext(self.api, jid, run.cancel)
        try:
            handler = self.handlers.get(job["kind"])
            out = (
                handler(ctx, job, self.s)
                if handler
                else Outcome("failed", error=f"tipo de trabajo desconocido: {job['kind']}")
            )
        except Exception as e:  # un trabajo defectuoso no tumba al worker
            log.exception("el trabajo %s falló", jid)
            out = Outcome("failed", error=f"{type(e).__name__}: {e}", log=ctx.log)
        for attempt in range(5):  # el resultado no se puede perder por un corte de red breve
            try:
                self.api.finish(jid, out.status, out.result, out.error, out.log)
                break
            except ApiError as e:
                log.error("la API rechazó el cierre del trabajo %s: %s", jid, e)
                break
            except Exception as e:
                log.warning("no se pudo cerrar el trabajo %s (intento %s): %s", jid, attempt + 1, e)
                time.sleep(2**attempt)
        with self.lock:
            self.running.pop(jid, None)
        log.info("trabajo %s (%s) terminó: %s", jid, job["kind"], out.status)

    def _start(self, job: dict) -> None:
        cancel = threading.Event()
        run = Running(job["kind"], cancel, threading.Thread())
        run.thread = threading.Thread(target=self._run_job, args=(job, run), name=f"job-{job['id']}", daemon=True)
        with self.lock:
            self.running[job["id"]] = run
        run.thread.start()

    def serve(self) -> None:
        delay = 1.0
        while not self.stop.is_set():
            try:
                if self.worker_id is None:
                    self.register()
                    threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True).start()
                kinds = self.free_kinds()
                if not kinds:
                    self.stop.wait(1.0)
                    continue
                job = self.api.claim(kinds, self.s.poll_wait)
                delay = 1.0
                if not job:
                    self.stop.wait(
                        0.2
                    )  # la API retiene la petición; esto solo evita un bucle vertiginoso si no lo hiciera
                if job:
                    log.info("trabajo %s (%s) recibido", job["id"], job["kind"])
                    self._start(job)
            except ApiError as e:
                if e.status in (401, 403):
                    log.error("la API rechazó el token del worker: %s", e)
                    self.stop.wait(30)
                else:
                    log.warning("error de la API: %s", e)
                    self.stop.wait(delay)
                    delay = min(30.0, delay * 2)
            except Exception as e:
                log.warning("sin conexión con la API (%s); reintento en %.0f s", e, delay)
                self.stop.wait(delay)
                delay = min(30.0, delay * 2)

    def shutdown(self, wait: float = 15.0) -> None:
        """Detiene el bucle y cancela lo que corre (la API verá los trabajos como cancelados)."""
        self.stop.set()
        with self.lock:
            runs = list(self.running.values())
        for r in runs:
            r.cancel.set()
        for r in runs:
            r.thread.join(wait)
