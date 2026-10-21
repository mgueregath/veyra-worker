"""Código de los procedimientos del espacio de trabajo: repositorio git, manifiesto y carga de funciones.

Lo usan `veyra-proc` (procesos por lotes) y `live_yolo` (flujos en vivo). Corre dentro de contenedores del espacio, donde el
repositorio del proyecto está en /workspace/project. Se ejecuta el código de un COMMIT concreto (no el árbol de trabajo).
"""

import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path("/workspace/project")


class ProcError(RuntimeError):
    pass


def git(*args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), "-c", "safe.directory=*", *args], capture_output=True, **kw)


def resolve(ref: str) -> str:
    if ref.startswith("-"):
        raise ProcError(f"referencia inválida: {ref}")
    r = git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", text=True)
    if r.returncode != 0:
        raise ProcError(
            f"no existe la versión «{ref}» en el repositorio del proyecto (¿falta hacer commit o la etiqueta?)"
        )
    return r.stdout.strip()


def checkout(sha: str) -> Path:
    """Extrae el árbol de ese commit en una carpeta temporal (el volumen del proyecto es de solo lectura)."""
    dest = Path(tempfile.mkdtemp(prefix="veyra-code-"))
    arc = git("archive", sha)
    if arc.returncode != 0:
        raise ProcError("no se pudo leer el repositorio: " + arc.stderr.decode()[-300:])
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arc.stdout, check=True)
    return dest


def manifest(code: Path) -> tuple[list[dict], str | None]:
    for name in ("veyra.yaml", "veyra.yml", "veyra.json"):
        f = code / name
        if not f.is_file():
            continue
        try:
            if name.endswith(".json"):
                data = json.loads(f.read_text())
            else:
                import yaml

                data = yaml.safe_load(f.read_text())
        except Exception as e:
            return [], f"{name} no se puede leer: {e}"
        procs = (data or {}).get("procedures", [])
        if not isinstance(procs, list):
            return [], f"{name}: «procedures» debe ser una lista"
        out = []
        for p in procs:
            if not isinstance(p, dict) or not p.get("name") or ":" not in str(p.get("entry", "")):
                return [], f"{name}: cada procedimiento necesita «name» y «entry» (módulo:función)"
            out.append(
                {
                    "name": str(p["name"]),
                    "entry": str(p["entry"]),
                    "doc": str(p.get("doc", "")),
                    "params": p.get("params") or {},
                }
            )
        return out, None
    return [], "El proyecto no tiene veyra.yaml en su repositorio: declara ahí sus procedimientos"


def load(name: str, ref: str = "HEAD", code: Path | None = None, sha: str | None = None):
    """Función del procedimiento y sus parámetros por defecto: (función, params, commit)."""
    sha = sha or resolve(ref)
    code = code or checkout(sha)
    procs, err = manifest(code)
    proc = next((p for p in procs if p["name"] == name), None)
    if proc is None:
        raise ProcError(f"el procedimiento «{name}» no está declarado en esa versión ({err or 'revise veyra.yaml'})")
    for extra in (code / "src", code):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    os.chdir(code)  # por si la función lee archivos del repositorio con rutas relativas
    mod, _, fn = proc["entry"].partition(":")
    return getattr(importlib.import_module(mod), fn), dict(proc["params"]), sha
