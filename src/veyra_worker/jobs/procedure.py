"""Procedimientos del espacio de trabajo: funciones de visión del repositorio git del proyecto.

Cada ejecución corre en un contenedor desechable de la imagen del espacio, SIN red, con el repositorio (volumen del proyecto) y
los datos en solo lectura, y solo con permiso de escritura en su carpeta de resultados. Se ejecuta el código de un commit
concreto, así que el resultado es reproducible. Los parámetros de la API se validan; el comando lo arma este módulo.
"""

import json
import re
import subprocess
import time
from pathlib import Path

from .. import paths
from ..config import Settings
from ..proc import run_streaming
from . import Outcome
from .context import JobContext
from .workspace import SLUG, WorkspaceError, _docker

NAME = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
REF = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./~^\-]{0,99}$")  # no puede empezar por «-» (sería una opción de git)


def _volume_exists(s: Settings, slug: str) -> bool:
    return _docker(s, "volume", "inspect", f"veyra-ws-{slug}").returncode == 0


def _base(s: Settings, slug: str, *, net_none: bool = True) -> list[str]:
    cmd = [
        s.docker_bin,
        "run",
        "--rm",
        "--user",
        "1000:3000",
        "--pids-limit",
        "1024",
        "-v",
        f"veyra-ws-{slug}:/workspace:ro",
    ]
    if net_none:
        cmd += ["--network", "none"]
    return cmd


def run_list(s: Settings, slug: str) -> Outcome:
    if not _volume_exists(s, slug):
        return Outcome(
            "done",
            result={
                "procedures": [],
                "ref": None,
                "error": "El proyecto aún no tiene espacio de trabajo: enciéndelo una vez",
            },
        )
    cmd = _base(s, slug) + ["--cpus", "1", "--memory", "1g", "--entrypoint", "veyra-proc", s.workspace_image, "list"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if r.returncode != 0:
        return Outcome("done", result={"procedures": [], "ref": None, "error": (r.stderr or r.stdout).strip()[-300:]})
    try:
        return Outcome("done", result=json.loads(r.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return Outcome("failed", error="respuesta inválida de veyra-proc: " + r.stdout[-200:])


def run(ctx: JobContext, job: dict, s: Settings) -> Outcome:
    p = job["payload"]
    try:
        slug = str(p.get("slug", ""))
        if not SLUG.match(slug):
            raise WorkspaceError(f"proyecto inválido: {slug!r}")
        action = p.get("action")
        if action == "list":
            return run_list(s, slug)
        if action != "run":
            raise WorkspaceError(f"acción desconocida: {action!r}")
        name, ref = str(p.get("name", "")), str(p.get("ref") or "HEAD")
        if not NAME.match(name) or not REF.match(ref):
            raise WorkspaceError("nombre o versión del procedimiento inválidos")
        out = paths.inside(s.data_root, p["out_dir"])
        images = [str(paths.inside(s.data_root, i)) for i in p["images"]]
        params = p.get("params") or {}
        if not isinstance(params, dict):
            raise WorkspaceError("los parámetros deben ser un objeto")
        rel = out.relative_to(s.data_root.resolve())
    except (WorkspaceError, paths.PathError, KeyError, ValueError) as e:
        return Outcome("failed", error=str(e))
    if not _volume_exists(s, slug):
        return Outcome("failed", error="el proyecto no tiene espacio de trabajo: enciéndelo al menos una vez")
    out.mkdir(parents=True, exist_ok=True)
    root = s.data_root.resolve()
    in_container = [
        f"/data/{Path(i).relative_to(root)}" for i in images
    ]  # dentro del contenedor la carpeta compartida es /data
    (out / "images.json").write_text(json.dumps(in_container))
    (out / "params.json").write_text(json.dumps(params))
    cmd = _base(s, slug) + [
        "--cpus", str(p.get("cpus", 4)), "--memory", str(p.get("memory", "8g")),
        "--mount", f"type=volume,src={s.data_volume},dst=/out,volume-subpath={rel}",
    ]  # fmt: skip
    if s.data_volume:
        cmd += ["-v", f"{s.data_volume}:/data:ro"]
    if p.get("gpu", True):
        cmd += ["--device", "nvidia.com/gpu=all"]
    cmd += ["--entrypoint", "veyra-proc", s.workspace_image, "run", "--name", name, "--ref", ref,
            "--images", "/out/images.json", "--params", "/out/params.json", "--out", "/out/result.json"]  # fmt: skip
    t0 = time.time()
    code, cancelled = run_streaming(cmd, lambda line: (ctx.add_log(line), ctx.progress()), ctx.cancel)
    if cancelled:
        return Outcome("cancelled", log=ctx.log)
    if code == 0 and (out / "result.json").exists():
        data = json.loads((out / "result.json").read_text())  # las rutas vuelven a ser las del worker
        data["results"] = {str(root / Path(k).relative_to("/data")): v for k, v in data["results"].items()}
        (out / "result.json").write_text(json.dumps(data))
        return Outcome("done", result={"images": len(images), "seconds": round(time.time() - t0, 1)}, log=ctx.log)
    return Outcome("failed", error=f"el procedimiento «{name}» falló (código {code}): " + ctx.log[-600:], log=ctx.log)
