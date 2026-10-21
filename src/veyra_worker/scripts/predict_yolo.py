#!/usr/bin/env python3
"""Predice cajas con un modelo YOLO para una lista de imágenes. Proceso aparte (ver train_yolo.py).

Entrada: JSON con la lista de rutas. Salida: JSON {ruta: [[x, y, w, h, puntaje, indice_de_clase], ...]} en píxeles de
la imagen original. Confianza mínima baja a propósito: para medir mAP hace falta la curva completa.
"""

import argparse
import json
import os
import sys

# PyTorch >= 2.6 carga los checkpoints con weights_only=True y Ultralytics 8.2 guarda la clase del modelo en ellos.
# Los pesos que se cargan aquí son archivos locales de confianza (preentrenados elegidos por el administrador o entrenados
# por esta plataforma): NO se debe apuntar este proceso a pesos de origen desconocido.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, help="archivo JSON con la lista de rutas")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    from ultralytics import YOLO

    paths = json.load(open(a.images))
    model = YOLO(a.weights)
    out = {}
    for i in range(0, len(paths), 16):
        for p, r in zip(
            paths[i : i + 16],
            model.predict(
                paths[i : i + 16], imgsz=a.imgsz, conf=a.conf, iou=0.6, device=a.device, verbose=False, max_det=100
            ),
            strict=True,
        ):
            out[p] = [
                [*(float(v) for v in (b[0], b[1], b[2] - b[0], b[3] - b[1])), float(c), int(k)]
                for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(), strict=True)
            ]
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    sys.exit(main())
