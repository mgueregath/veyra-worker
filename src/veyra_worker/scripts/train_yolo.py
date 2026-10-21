#!/usr/bin/env python3
"""Entrena un detector YOLO (Ultralytics). Proceso aparte: lo lanza la API con el Python que tenga PyTorch.

No importa nada de `app`: solo recibe argumentos y escribe líneas `PROGRESS {...}` y, al final, `RESULT {...}`.
AVISO DE LICENCIA: Ultralytics es AGPL-3.0; si hay que evitarlo, este archivo es lo único que cambia (otro backend con
la misma interfaz de línea de comandos).
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# PyTorch >= 2.6 carga los checkpoints con weights_only=True y Ultralytics 8.2 guarda la clase del modelo en ellos.
# Los pesos que se cargan aquí son archivos locales de confianza (preentrenados elegidos por el administrador o entrenados
# por esta plataforma): NO se debe apuntar este proceso a pesos de origen desconocido.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True, help="ruta a pesos .pt o nombre (se descarga)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default="train")
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="dónde dejar best.pt")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.weights)

    def on_epoch(trainer):
        m = {k: float(v) for k, v in (trainer.metrics or {}).items() if isinstance(v, (int, float))}
        print(
            "PROGRESS " + json.dumps({"epoch": trainer.epoch + 1, "epochs": trainer.epochs, "metrics": m}), flush=True
        )

    model.add_callback("on_fit_epoch_end", on_epoch)
    # aumentos pensados para insectos vistos desde arriba: giros y rotaciones son válidos, no se deforma la forma
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, project=a.project, name=a.name, device=a.device,
        seed=a.seed, exist_ok=True, plots=False, workers=2, degrees=15, scale=0.3, fliplr=0.5, flipud=0.5, mosaic=1.0,
        patience=max(10, a.epochs // 3), verbose=False,
    )  # fmt: skip
    save = Path(a.project) / a.name
    best = save / "weights" / "best.pt"
    if not best.exists():
        best = save / "weights" / "last.pt"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, a.out)
    # métricas de la validación con los mejores pesos
    val = YOLO(a.out).val(
        data=a.data, imgsz=a.imgsz, batch=a.batch, device=a.device, plots=False, verbose=False, split="val"
    )
    res = {
        "map50": float(val.box.map50), "map50_95": float(val.box.map), "precision": float(val.box.mp), "recall": float(val.box.mr),
        "weights": a.out,
    }  # fmt: skip
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    sys.exit(main())
