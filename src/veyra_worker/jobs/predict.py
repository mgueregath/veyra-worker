"""Predicción por lotes (preanotar, evaluar): scripts/predict_yolo.py; el resultado se escribe en la carpeta compartida."""

import json
import os
import tempfile
from pathlib import Path

from .. import paths
from ..config import Settings
from ..proc import run_streaming
from . import Outcome
from .context import JobContext


def run(ctx: JobContext, job: dict, s: Settings) -> Outcome:
    a = job["payload"]
    root = s.data_root
    images = [str(paths.inside(root, p)) for p in a["images"]]
    out = paths.inside(root, a["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        lst = Path(tmp) / "images.json"
        lst.write_text(json.dumps(images))
        cmd = [
            s.python, str(s.scripts_dir / "predict_yolo.py"),
            "--weights", paths.weights(root, s.pretrained_dirs, a["weights"]),
            "--images", str(lst), "--out", str(out),
            "--imgsz", str(int(a.get("imgsz", 1280))), "--conf", str(float(a.get("conf", 0.001))),
            "--device", s.device,
        ]  # fmt: skip
        code, cancelled = run_streaming(
            cmd, lambda line: (ctx.add_log(line), ctx.progress()), ctx.cancel, env={**os.environ}
        )
    if cancelled:
        return Outcome("cancelled", log=ctx.log)
    if code == 0 and out.exists():
        return Outcome("done", result={"out": str(out), "images": len(images)}, log=ctx.log)
    return Outcome("failed", error=f"falló la inferencia (código {code})", log=ctx.log)
