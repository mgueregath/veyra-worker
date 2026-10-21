"""Entrenamiento YOLO: ejecuta scripts/train_yolo.py con el mismo contrato de siempre (líneas PROGRESS y RESULT)."""

import json
import os

from .. import paths
from ..config import Settings
from ..proc import run_streaming
from . import Outcome
from .context import JobContext


def build_cmd(job: dict, s: Settings) -> list[str]:
    a = job["payload"]
    root = s.data_root
    return [
        s.python,
        str(s.scripts_dir / "train_yolo.py"),
        "--data", str(paths.inside(root, a["data"])),
        "--weights", paths.weights(root, s.pretrained_dirs, a["weights"]),
        "--epochs", str(int(a["epochs"])),
        "--imgsz", str(int(a["imgsz"])),
        "--batch", str(int(a["batch"])),
        "--project", str(paths.inside(root, a["project"])),
        "--name", str(a.get("name", "train")).replace("/", "_"),
        "--device", s.device,
        "--seed", str(int(a.get("seed", 0))),
        "--out", str(paths.inside(root, a["out"])),
    ]  # fmt: skip


def run(ctx: JobContext, job: dict, s: Settings) -> Outcome:
    cmd = build_cmd(job, s)
    state: dict = {"result": None}

    def on_line(line: str) -> None:
        if line.startswith("PROGRESS "):
            ctx.progress(progress=json.loads(line[9:]))
        elif line.startswith("RESULT "):
            state["result"] = json.loads(line[7:])
        else:
            ctx.add_log(line)
            ctx.progress()

    code, cancelled = run_streaming(cmd, on_line, ctx.cancel, env={**os.environ})
    if cancelled:
        return Outcome("cancelled", log=ctx.log)
    if code == 0 and state["result"]:
        return Outcome("done", result=state["result"], log=ctx.log)
    return Outcome("failed", error=f"el proceso terminó con código {code}", log=ctx.log)
