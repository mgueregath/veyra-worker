#!/usr/bin/env python3
"""Proceso de detección en vivo: lee una fuente (archivo, RTSP, MJPEG), detecta, sigue y sirve el resultado por HTTP local.

Lo lanza la API (una sesión = un proceso) con el Python que tenga PyTorch. Escucha SOLO en 127.0.0.1 y exige un token:
la API es quien autentica a las personas y reenvía.

  GET  /stream.mjpg?token=…   video con las cajas dibujadas (multipart/x-mixed-replace)
  GET  /stats?token=…         JSON: fps, latencia, recuentos, umbral, estado de la fuente
  GET  /frame.jpg?token=…     el último fotograma SIN dibujar; cabecera X-Detections con las cajas (JSON)
  POST /control?token=…       {"threshold": 0.4, "paused": false, "record": true}
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from live_core import FakeDetector, IouTracker, LatestFrameReader, NullDetector, Stats, YoloDetector, draw_overlay  # noqa: E402


class Shared:
    """Estado compartido entre el bucle de detección y el servidor HTTP."""

    def __init__(self, names, threshold):
        self.cond = threading.Condition()
        self.jpeg = b""
        self.raw_frame = None  # último fotograma SIN dibujar (se codifica solo si alguien lo pide)
        self.dets: list = []
        self.version = 0
        self.names = names
        self.threshold = threshold
        self.paused = False
        self.recording = False
        self.record_path = ""
        self.show_boxes = True
        self.show_conf = True
        self.show_classes: set[int] | None = None  # None = todas
        self.stats: dict = {"frames": 0, "fps": 0.0, "latency_ms": 0.0}
        self.stop = threading.Event()


def _view(sh: Shared) -> dict:
    return {
        "threshold": sh.threshold,
        "paused": sh.paused,
        "recording": sh.recording,
        "show_boxes": sh.show_boxes,
        "show_conf": sh.show_conf,
        "show_classes": None if sh.show_classes is None else sorted(sh.show_classes),
    }


def make_handler(sh: Shared, token: str):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _ok(self):
            q = parse_qs(urlparse(self.path).query)
            if (q.get("token") or [""])[0] != token and self.headers.get("X-Token") != token:
                self.send_error(403, "token inválido")
                return None
            return urlparse(self.path).path

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            path = self._ok()
            if path is None:
                return
            if path == "/stats":
                return self._json(sh.stats)
            if path == "/frame.jpg":
                with sh.cond:
                    frame, dets = sh.raw_frame, sh.dets
                if frame is None:
                    return self.send_error(503, "todavía no hay fotograma")
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                raw = buf.tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("X-Detections", json.dumps([[*d[:4], d[4], d[5]] for d in dets]))
                self.end_headers()
                self.wfile.write(raw)
                return
            if path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seen = -1
                try:
                    while not sh.stop.is_set():
                        with sh.cond:
                            # hasta que exista el primer fotograma no se envía nada (un JPEG vacío rompería la imagen)
                            sh.cond.wait_for(
                                lambda seen=seen: (sh.version != seen and sh.jpeg) or sh.stop.is_set(), timeout=2
                            )
                            if sh.version == seen or not sh.jpeg:
                                continue
                            seen, jpg = sh.version, sh.jpeg
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg)
                            + jpg
                            + b"\r\n"
                        )
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_error(404)

        def do_POST(self):
            if self._ok() != "/control":
                return self.send_error(404)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            if "threshold" in body:
                sh.threshold = min(1.0, max(0.0, float(body["threshold"])))
            if "paused" in body:
                sh.paused = bool(body["paused"])
            if "record" in body:
                sh.recording = bool(body["record"])
            if "show_boxes" in body:
                sh.show_boxes = bool(body["show_boxes"])
            if "show_conf" in body:
                sh.show_conf = bool(body["show_conf"])
            if "show_classes" in body:
                sh.show_classes = None if body["show_classes"] is None else {int(k) for k in body["show_classes"]}
            self._json(_view(sh))

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", help="pesos .pt (si se omite, usar --detector)")
    ap.add_argument("--detector", choices=["yolo", "fake", "none"], default="yolo")
    ap.add_argument("--classes", default="", help="JSON con los nombres de clase (si no, los del modelo)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument(
        "--host", default="127.0.0.1", help="dirección donde escucha (el worker usa 0.0.0.0 para que la API lo alcance)"
    )
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--loop", action="store_true", help="repetir el archivo al llegar al final")
    ap.add_argument("--no-realtime", action="store_true", help="archivo: leer lo más rápido posible (pruebas)")
    ap.add_argument("--record-dir", default="")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--stream-width", type=int, default=1280, help="ancho máximo del flujo MJPEG (0 = original)")
    a = ap.parse_args()

    det = (
        YoloDetector(a.weights, a.imgsz, a.device)
        if a.detector == "yolo" and a.weights
        else FakeDetector()
        if a.detector == "fake"
        else NullDetector()
    )
    names = json.loads(a.classes) if a.classes else getattr(det, "names", ["objeto"])
    sh = Shared(names, a.threshold)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(sh, a.token))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    reader = LatestFrameReader(a.source, realtime=not a.no_realtime, loop=a.loop)
    tracker, stats, writer = IouTracker(), Stats(), None
    sh.stats = {
        "frames": 0,
        "fps": 0.0,
        "latency_ms": 0.0,
        **_source_stats(reader, tracker, sh),
    }  # completas desde el primer momento
    print(f"LISTO {a.port}", flush=True)
    try:
        while not sh.stop.is_set():
            got = reader.read_new(timeout=1.0)
            if got is None:
                sh.stats = {**sh.stats, **_source_stats(reader, tracker, sh)}
                if reader.ended:
                    break
                continue
            frame, idx, ts = got
            if sh.paused:  # pausado no procesa, pero el estado (y la fuente) se siguen informando
                sh.stats = {**sh.stats, **_source_stats(reader, tracker, sh)}
                continue
            dets = [d for d in det(frame) if d[4] >= sh.threshold]
            tracks = tracker.update(dets)
            stats.add(ts)
            vis = [t for t in tracks if sh.show_boxes and (sh.show_classes is None or t.cls in sh.show_classes)]
            shown = draw_overlay(
                frame, vis, names, "", sh.show_conf
            )  # velocidad y latencia se muestran fuera del video
            if (
                a.stream_width and shown.shape[1] > a.stream_width
            ):  # el flujo para el navegador va reducido: codificar 1080p es lo que más cuesta
                shown = cv2.resize(
                    shown,
                    (a.stream_width, round(shown.shape[0] * a.stream_width / shown.shape[1])),
                    interpolation=cv2.INTER_AREA,
                )
            ok1, jpg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
            if sh.recording and a.record_dir:
                if writer is None:
                    sh.record_path = str(Path(a.record_dir) / f"clip_{int(time.time())}.mp4")
                    Path(a.record_dir).mkdir(parents=True, exist_ok=True)
                    fps = (
                        reader.source_fps or stats.fps or 15.0
                    )  # el de la fuente, acotado a un rango que el escritor acepte
                    writer = cv2.VideoWriter(
                        sh.record_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        min(60.0, max(5.0, fps)),
                        (frame.shape[1], frame.shape[0]),
                    )
                    if not writer.isOpened():  # no se pudo grabar: se avisa en las estadísticas
                        sh.record_path, sh.recording, writer = "", False, None
                        sh.stats = {**sh.stats, "record_error": "no se pudo abrir el escritor de video"}
                if writer is not None:
                    writer.write(shown)
            elif writer is not None:
                writer.release()
                writer = None
            with sh.cond:
                if ok1:
                    sh.jpeg, sh.raw_frame, sh.dets = jpg.tobytes(), frame, dets
                    sh.version += 1
                sh.stats = {
                    "frames": stats.frames,
                    "fps": round(stats.fps, 2),
                    "latency_ms": round(stats.latency_ms, 1),
                    "frame": idx,
                    **_source_stats(reader, tracker, sh),
                }
                sh.cond.notify_all()
    finally:
        sh.stop.set()
        with sh.cond:
            sh.cond.notify_all()
        if writer is not None:
            writer.release()
        reader.close()
        srv.shutdown()


def _source_stats(reader, tracker, sh):
    return {
        "source_fps": round(reader.source_fps, 2),
        "captured": reader.captured,
        "dropped": reader.dropped,
        "reconnects": reader.reconnects,
        "open_failures": reader.open_failures,
        "ended": reader.ended,
        "error": reader.last_error if not reader.captured or reader.reconnects else "",
        "size": list(reader.size) if reader.size else None,
        "threshold": sh.threshold,
        "paused": sh.paused,
        "recording": sh.recording,
        "show_boxes": sh.show_boxes,
        "show_conf": sh.show_conf,
        "show_classes": None if sh.show_classes is None else sorted(sh.show_classes),
        "record_path": sh.record_path,
        "totals": {sh.names[k] if k < len(sh.names) else str(k): v for k, v in tracker.totals.items()},
        "active": len([t for t in tracker.tracks if t.confirmed and t.misses == 0]),
        "classes": sh.names,
        "pid": os.getpid(),
    }


if __name__ == "__main__":
    sys.exit(main())
