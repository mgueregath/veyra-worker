#!/usr/bin/env python3
"""Prueba real con Docker de los procedimientos del espacio de trabajo: listar, ejecutar, versión fija, errores y aislamiento.

    python scripts/e2e_procedure.py [imagen]        (requiere la imagen veyra-workspace construida)
Usa un volumen de datos temporal enlazado a una carpeta temporal: no toca el NFS ni proyectos reales.
"""

import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from veyra_worker.config import Settings  # noqa: E402
from veyra_worker.jobs import procedure as pr  # noqa: E402
from veyra_worker.jobs.context import JobContext  # noqa: E402

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "docker.labs.hapticus.cl/veyra-workspace:0.2.0"
SLUG, DVOL = "e2eproc", "veyra-e2e-data"
tmp = Path(tempfile.mkdtemp(prefix="procE2E_"))
res = {}


def check(name, ok, extra=""):
    print(("OK    " if ok else "FALLA ") + name + "  " + str(extra)[:220])
    res[name] = bool(ok)


def sh(*cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


class Api:
    def progress(self, *_):
        return {}


S = Settings(api_url="http://x", token="t", data_root=tmp, workspaces=True, data_volume=DVOL, workspace_image=IMAGE)


def job(action, **kw):
    return JobContext(Api(), 1, threading.Event()), {"id": 1, "payload": {"action": action, "slug": SLUG, **kw}}


def workspace_sh(script: str):
    """Ejecuta un script en el volumen del proyecto como usuario veyra (para preparar su repositorio)."""
    r = sh(
        "docker",
        "run",
        "--rm",
        "--user",
        "1000:3000",
        "-v",
        f"veyra-ws-{SLUG}:/workspace",
        "--entrypoint",
        "sh",
        IMAGE,
        "-c",
        script,
    )
    return r.stdout + r.stderr


def run_proc(name, ref="HEAD", params=None, n=2):
    out = tmp / "runs" / f"o{len(list((tmp / 'runs').glob('o*')))}"
    imgs = [str(tmp / "media" / f"i{i}.jpg") for i in range(n)]
    ctx, j = job("run", name=name, ref=ref, params=params or {}, images=imgs, out_dir=str(out))
    o = pr.run(ctx, j, S)
    data = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else None
    return o, data


try:
    for d in ("media", "runs"):
        (tmp / d).mkdir()
    sh("docker", "volume", "rm", "-f", DVOL)
    sh(
        "docker",
        "volume",
        "create",
        "--driver",
        "local",
        "--opt",
        "type=none",
        "--opt",
        "o=bind",
        "--opt",
        f"device={tmp}",
        DVOL,
    )
    sh("docker", "volume", "rm", "-f", f"veyra-ws-{SLUG}")
    sh("docker", "volume", "create", f"veyra-ws-{SLUG}")
    # lo que hace el arranque del espacio (con el volumen vacío Docker restaura los permisos de la imagen: hay que dejar contenido)
    sh(
        "docker",
        "run",
        "--rm",
        "-v",
        f"veyra-ws-{SLUG}:/workspace",
        "--entrypoint",
        "sh",
        IMAGE,
        "-c",
        "chown 1000:3000 /workspace && mkdir -p /workspace/.veyra && chown 1000:3000 /workspace/.veyra",
    )
    import cv2
    import numpy as np

    for i in range(2):
        cv2.imwrite(str(tmp / "media" / f"i{i}.jpg"), np.full((60, 100, 3), 100 + 50 * i, np.uint8))
    ctx, j = job("list")
    o = pr.run(ctx, j, S)
    check(
        "sin repositorio ni manifiesto explica qué falta (no falla)",
        o.status == "done" and o.result["procedures"] == [] and o.result["error"],
        o.result,
    )
    workspace_sh("""
mkdir -p /workspace/project/src/mis_filtros && cd /workspace/project && git init -q -b main && git config user.email t@t && git config user.name t
cat > veyra.yaml <<'Y'
procedures:
  - name: cajita
    entry: mis_filtros.uno:cajita
    doc: Devuelve una caja de tamaño configurable
    params: {lado: 10, score: 0.8}
  - name: roto
    entry: mis_filtros.uno:roto
  - name: red
    entry: mis_filtros.uno:red
Y
touch src/mis_filtros/__init__.py
cat > src/mis_filtros/uno.py <<'P'
def cajita(img, lado=10, score=0.8):
    h, w = img.shape[:2]
    return [{"bbox": [5, 5, lado, lado], "score": score, "label": "louse"}, {"bbox": [0, 0, 0, 5], "score": 1}]  # la segunda es inválida
def roto(img):
    raise ValueError("falla a propósito")
def red(img):
    import socket
    socket.create_connection(("1.1.1.1", 53), timeout=3)
    return []
P
git add -A && git commit -q -m v1 && git tag v1
""")
    ctx, j = job("list")
    o = pr.run(ctx, j, S)
    names = [p["name"] for p in o.result["procedures"]]
    check(
        "lista los procedimientos que declara el repositorio del proyecto",
        names == ["cajita", "roto", "red"]
        and o.result["ref"]
        and o.result["procedures"][0]["params"] == {"lado": 10, "score": 0.8},
        o.result,
    )
    o, data = run_proc("cajita")
    r0 = data and data["results"][str(tmp / "media" / "i0.jpg")]
    check(
        "ejecuta el procedimiento sobre las imágenes y devuelve cajas (descarta las inválidas)",
        o.status == "done" and r0 == [{"bbox": [5, 5, 10, 10], "score": 0.8, "label": "louse"}],
        (o.error, o.log, data),
    )
    check("registra el commit exacto que se ejecutó", data and len(data["ref"]) == 40, data and data["ref"])
    o, data = run_proc("cajita", params={"lado": 30, "score": 0.5})
    check(
        "los parámetros reemplazan a los del manifiesto",
        data
        and data["results"][str(tmp / "media" / "i0.jpg")][0]["bbox"][2] == 30
        and data["results"][str(tmp / "media" / "i0.jpg")][0]["score"] == 0.5,
    )
    workspace_sh(
        "cd /workspace/project && sed -i 's/\\[5, 5, lado, lado\\]/[7, 7, lado, lado]/' src/mis_filtros/uno.py && git commit -qam v2 && echo ok"
    )
    o2, d2 = run_proc("cajita")
    o1, d1 = run_proc("cajita", ref="v1")
    check(
        "HEAD usa el último commit y una etiqueta fija la versión anterior",
        d2["results"][str(tmp / "media" / "i0.jpg")][0]["bbox"][0] == 7
        and d1["results"][str(tmp / "media" / "i0.jpg")][0]["bbox"][0] == 5
        and d1["ref"] != d2["ref"],
    )
    workspace_sh(
        "cd /workspace/project && sed -i 's/\\[7, 7, lado, lado\\]/[9, 9, lado, lado]/' src/mis_filtros/uno.py"
    )  # cambio sin commit
    o3, d3 = run_proc("cajita")
    check(
        "lo que no está confirmado en git no se ejecuta (reproducible)",
        d3["results"][str(tmp / "media" / "i0.jpg")][0]["bbox"][0] == 7,
    )
    o, data = run_proc("cajita", ref="no-existe")
    check(
        "una versión inexistente falla con un mensaje claro", o.status == "failed" and "no-existe" in o.error, o.error
    )
    o, data = run_proc("roto")
    check(
        "un error del procedimiento se informa con su traza",
        o.status == "failed" and "falla a propósito" in o.error,
        o.error,
    )
    o, data = run_proc("fantasma")
    check("un procedimiento no declarado se rechaza", o.status == "failed" and "fantasma" in o.error, o.error)
    o, data = run_proc("red")
    check(
        "el procedimiento no tiene red",
        o.status == "failed"
        and ("unreachable" in o.error.lower() or "network" in o.error.lower() or "Errno" in o.error),
        o.error,
    )
    o, data = run_proc("cajita", ref="v1", n=1)
    r = sh(
        "docker",
        "run",
        "--rm",
        "--user",
        "1000:3000",
        "-v",
        f"veyra-ws-{SLUG}:/workspace:ro",
        "--entrypoint",
        "sh",
        IMAGE,
        "-c",
        "touch /workspace/x 2>&1; echo $?",
    )
    check(
        "el volumen del proyecto es de solo lectura dentro del procedimiento",
        "Read-only" in r.stdout or "1" in r.stdout.strip().splitlines()[-1],
        r.stdout,
    )
    print(f"\nRESUMEN: {sum(res.values())}/{len(res)} OK")
finally:
    sh("docker", "volume", "rm", "-f", f"veyra-ws-{SLUG}")
    sh("docker", "volume", "rm", "-f", DVOL)
sys.exit(0 if res and all(res.values()) else 1)
