"""Imita workers/predict_yolo.py: devuelve siempre un acierto exacto sobre (10,10,20,20) y un falso positivo."""

import argparse
import json

ap = argparse.ArgumentParser()
for a in ("weights", "images", "out", "device"):
    ap.add_argument("--" + a)
ap.add_argument("--imgsz", type=int)
ap.add_argument("--conf", type=float, default=0.001)
a = ap.parse_args()
json.dump({p: [[10, 10, 20, 20, 0.9, 0], [70, 60, 8, 8, 0.2, 0]] for p in json.load(open(a.images))}, open(a.out, "w"))
