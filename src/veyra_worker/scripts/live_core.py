"""Núcleo del motor de inferencia en vivo. Sin PyTorch: solo OpenCV y NumPy, así se puede probar con cualquier detector.

- LatestFrameReader: lee una fuente (archivo, RTSP o MJPEG por HTTP) en un hilo y entrega SIEMPRE el fotograma más reciente.
  Si el detector va más lento que la fuente, los fotogramas viejos se descartan (la latencia no crece); si la conexión se
  cae, reconecta con espera creciente.
- IouTracker: sigue las cajas entre fotogramas para contar cada objeto una sola vez.
- Stats / draw_overlay: velocidad, latencia, recuentos y dibujo de cajas.
"""

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

Det = tuple[float, float, float, float, float, int]  # x, y, w, h, puntaje, índice de clase


def is_network(source: str) -> bool:
    return source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://", "rtmp://", "udp://", "tcp://"))


def open_capture(source: str) -> cv2.VideoCapture:
    if source.lower().startswith("rtsp"):  # TCP evita los cortes y la corrupción de imagen típicos de UDP
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG)


class LatestFrameReader:
    """Hilo de captura que conserva solo el último fotograma."""

    def __init__(self, source: str, realtime: bool = True, loop: bool = False, max_backoff: float = 8.0):
        self.source, self.realtime, self.loop, self.max_backoff = source, realtime, loop, max_backoff
        self.network = is_network(source)
        self._frame: np.ndarray | None = None
        self._idx = 0  # número del último fotograma capturado
        self._taken = 0  # último que entregó read_new
        self._ts = 0.0
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self.ended = False  # el archivo terminó (sin loop)
        self.reconnects = 0
        self.open_failures = 0
        self.captured = 0
        self.delivered = 0
        self.source_fps = 0.0
        self.size: tuple[int, int] | None = None
        self.last_error = ""
        self._thread = threading.Thread(target=self._run, name="reader", daemon=True)
        self._thread.start()

    # ---- hilo ----
    def _run(self) -> None:
        backoff, t_start, media_start = 0.5, None, None
        while not self._stop.is_set():
            cap = open_capture(self.source)
            if not cap.isOpened():
                self.last_error = "no se pudo abrir la fuente"
                self.open_failures += 1
                self._wait_retry(backoff)
                backoff = min(self.max_backoff, backoff * 2)
                continue
            backoff, t_start, media_start = 0.5, time.time(), None
            self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if self.realtime and not self.network:  # archivo: se respeta el ritmo real del video
                    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    media_start = ms if media_start is None else media_start
                    wait = (ms - media_start) / 1000.0 - (time.time() - t_start)
                    if wait > 0 and self._stop.wait(wait):
                        break
                with self._cond:
                    self._frame, self._ts = frame, time.time()
                    self._idx += 1
                    self.captured += 1
                    self.size = (frame.shape[1], frame.shape[0])
                    self._cond.notify_all()
            cap.release()
            if self._stop.is_set():
                return
            if not self.network and not self.loop:  # fin del archivo
                with self._cond:
                    self.ended = True
                    self._cond.notify_all()
                return
            if self.network:
                self.reconnects += 1
                self.last_error = "se perdió la conexión: reconectando"
                self._wait_retry(backoff)
                backoff = min(self.max_backoff, backoff * 2)

    def _wait_retry(self, secs: float) -> None:
        self._stop.wait(secs)

    # ---- consumo ----
    def read_new(self, timeout: float = 1.0) -> tuple[np.ndarray, int, float] | None:
        """El fotograma más reciente que aún no se entregó, esperando hasta `timeout`. None si no llegó ninguno."""
        with self._cond:
            if self._idx == self._taken:
                self._cond.wait(timeout)
            if self._idx == self._taken or self._frame is None:
                return None
            self._taken = self._idx
            self.delivered += 1
            return self._frame, self._idx, self._ts

    @property
    def dropped(self) -> int:
        return max(0, self.captured - self.delivered)

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=3)


# ---------------------------------------------------------------- seguimiento
def iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax2, bx2) - max(a[0], b[0]), min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


@dataclass
class Track:
    id: int
    box: tuple[float, float, float, float]
    cls: int
    score: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False


@dataclass
class IouTracker:
    """Seguimiento por IoU (voraz). Un objeto «cuenta» cuando se confirma (`min_hits` fotogramas seguidos).

    AVISO: con la cámara en la mano el fondo se mueve y un mismo piojo puede perder su identidad: el total es una COTA
    SUPERIOR del número real de piojos distintos; sirve para comparar sesiones, no como conteo exacto.
    """

    iou_thr: float = 0.3
    min_hits: int = 3
    max_age: int = 10
    tracks: list[Track] = field(default_factory=list)
    _next: int = 1
    totals: dict[int, int] = field(default_factory=dict)  # clase → objetos distintos confirmados

    def update(self, dets: list[Det]) -> list[Track]:
        pairs = sorted(
            (
                (iou(t.box, d[:4]), ti, di)
                for ti, t in enumerate(self.tracks)
                for di, d in enumerate(dets)
                if t.cls == d[5]
            ),
            reverse=True,
        )
        used_t: set[int] = set()
        used_d: set[int] = set()
        for v, ti, di in pairs:
            if v < self.iou_thr:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t, d = self.tracks[ti], dets[di]
            t.box, t.score, t.hits, t.misses = tuple(d[:4]), d[4], t.hits + 1, 0
            if not t.confirmed and t.hits >= self.min_hits:
                t.confirmed = True
                self.totals[t.cls] = self.totals.get(t.cls, 0) + 1
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
                t.hits = 0 if not t.confirmed else t.hits  # no confirmada: debe ser consecutiva
        for di, d in enumerate(dets):
            if di not in used_d:
                t = Track(self._next, tuple(d[:4]), d[5], d[4])
                self._next += 1
                if self.min_hits <= 1:
                    t.confirmed = True
                    self.totals[t.cls] = self.totals.get(t.cls, 0) + 1
                self.tracks.append(t)
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age and (t.confirmed or t.misses == 0)]
        return [t for t in self.tracks if t.confirmed and t.misses == 0]


# ---------------------------------------------------------------- estadísticas y dibujo
class Stats:
    """Velocidad de procesamiento y latencia (de la captura al resultado) en una ventana móvil."""

    def __init__(self, window: int = 60):
        self.times: deque[float] = deque(maxlen=window)
        self.lat: deque[float] = deque(maxlen=window)
        self.frames = 0

    def add(self, captured_at: float) -> None:
        now = time.time()
        self.times.append(now)
        self.lat.append(now - captured_at)
        self.frames += 1

    @property
    def fps(self) -> float:
        return (
            (len(self.times) - 1) / (self.times[-1] - self.times[0])
            if len(self.times) >= 2 and self.times[-1] > self.times[0]
            else 0.0
        )

    @property
    def latency_ms(self) -> float:
        return 1000 * sum(self.lat) / len(self.lat) if self.lat else 0.0


PALETTE = [(80, 185, 63), (255, 140, 179), (73, 81, 248), (35, 167, 227), (240, 201, 76)]  # BGR


def draw_overlay(frame: np.ndarray, tracks, names: list[str], banner: str = "", show_conf: bool = True) -> np.ndarray:
    """Dibuja cajas, clase, puntaje e id sobre una COPIA del fotograma."""
    out = frame.copy()
    th = max(2, round(out.shape[1] / 600))
    for t in tracks:
        x, y, w, h = (int(round(v)) for v in t.box)
        col = PALETTE[t.cls % len(PALETTE)]
        cv2.rectangle(out, (x, y), (x + w, y + h), col, th)
        label = (
            f"{names[t.cls] if t.cls < len(names) else t.cls}" + (f" {t.score:.2f}" if show_conf else "") + f" #{t.id}"
        )
        (tw, tH), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (x, max(0, y - tH - 8)), (x + tw + 6, y), col, -1)
        cv2.putText(out, label, (x + 3, max(tH, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    if banner:
        cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(out, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- detectores
class FakeDetector:
    """Detector de pruebas: una caja que avanza 5 px por fotograma y otra quieta con puntaje bajo."""

    def __init__(self):
        self.n = 0

    def __call__(self, frame: np.ndarray) -> list[Det]:
        self.n += 1
        h, w = frame.shape[:2]
        x = min(w - 40, 10 + 5 * self.n)
        return [(x, h // 3, 30.0, 30.0, 0.9, 0), (w // 2, h // 2, 20.0, 20.0, 0.15, 0)]


class NullDetector:
    def __call__(self, frame: np.ndarray) -> list[Det]:
        return []


class YoloDetector:
    """YOLO (Ultralytics) en la GPU. Importa torch solo aquí, para que el resto del núcleo funcione sin él."""

    def __init__(self, weights: str, imgsz: int = 1280, device: str = "0", half: bool = True, conf: float = 0.05):
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # pesos locales de confianza (ver train_yolo.py)
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz, self.device, self.half, self.conf = imgsz, device, half and device != "cpu", conf
        self.names = [self.model.names[i] for i in sorted(self.model.names)]
        self.model.predict(
            np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, device=device, half=self.half, verbose=False
        )  # calentamiento

    def __call__(self, frame: np.ndarray) -> list[Det]:
        r = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=0.6,
            device=self.device,
            half=self.half,
            verbose=False,
            max_det=100,
        )[0]
        return [
            (b[0], b[1], b[2] - b[0], b[3] - b[1], c, int(k))
            for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(), strict=True)
        ]
