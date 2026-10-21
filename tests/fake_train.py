"""Imita workers/train_yolo.py (mismos argumentos y mismas líneas de salida) sin PyTorch, para probar la orquestación."""

import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
for a in ("data", "weights", "project", "name", "device", "out"):
    ap.add_argument("--" + a)
for a in ("epochs", "imgsz", "batch", "seed"):
    ap.add_argument("--" + a, type=int, default=0)
a = ap.parse_args()
assert Path(a.data).is_file(), "data.yaml no existe"
assert (Path(a.data).parent / "images" / "train").is_dir()
print("cargando datos…", flush=True)
if a.epochs == 7:  # fallo simulado
    print("Traceback: algo salió mal", flush=True)
    sys.exit(3)
for e in range(1, a.epochs + 1):
    print(
        "PROGRESS " + json.dumps({"epoch": e, "epochs": a.epochs, "metrics": {"metrics/mAP50(B)": 0.1 * e}}), flush=True
    )
    time.sleep(60 if a.epochs == 9 else 0.05)  # epochs == 9: tarda «mucho» (para probar la cancelación)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_bytes(b"pesos-falsos")
print(
    "RESULT " + json.dumps({"map50": 0.5, "map50_95": 0.3, "precision": 0.6, "recall": 0.7, "weights": a.out}),
    flush=True,
)
