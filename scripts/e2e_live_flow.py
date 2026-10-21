#!/usr/bin/env python3
"""Prueba real con Docker de una sesión en vivo con flujo (modelo YOLO O procedimiento del espacio de trabajo).

    python scripts/e2e_live_flow.py [imagen] [pesos.pt]      (requiere la imagen veyra-workspace y una GPU con Docker)
Usa un volumen de datos temporal enlazado a una carpeta temporal y un proyecto temporal: no toca el NFS ni proyectos reales.
"""

import json
import secrets
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from veyra_worker.config import Settings  # noqa: E402
from veyra_worker.jobs import live  # noqa: E402

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "docker.labs.hapticus.cl/veyra-workspace:0.4.0"
WEIGHTS = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/uach/DEV/lice/labelhub/backend/yolov8n.pt")
SLUG, DVOL = "e2elive", "veyra-e2e-live-data"
tmp = Path(tempfile.mkdtemp(prefix="liveE2E_"))
res = {}


def check(name, ok, extra=""):
    print(("OK    " if ok else "FALLA ") + name + "  " + str(extra)[:240])
    res[name] = bool(ok)


def sh(*cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def get(port, token, path):
    return urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}{'&' if '?' in path else '?'}token={token}", timeout=10
    )


S = Settings(api_url="http://x", token="t", data_root=tmp, data_volume=DVOL, workspace_image=IMAGE)
name, container = None, None
try:
    (tmp / "m").mkdir()
    (tmp / "m" / "best.pt").write_bytes(WEIGHTS.read_bytes())
    import cv2
    import numpy as np

    vw = cv2.VideoWriter(str(tmp / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(60):
        vw.write(np.full((240, 320, 3), 60 + i, np.uint8))
    vw.release()
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
    r = sh(
        "docker", "run", "--rm", "--user", "root", "-v", f"veyra-ws-{SLUG}:/workspace", "--entrypoint", "sh", IMAGE, "-c",
        "mkdir -p /workspace/.veyra && chown -R 1000:3000 /workspace",
    )  # fmt: skip
    r = sh(
        "docker", "run", "--rm", "--user", "1000:3000", "-v", f"veyra-ws-{SLUG}:/workspace", "--entrypoint", "sh", IMAGE, "-c",
        """
mkdir -p /workspace/project/src/mis_filtros && cd /workspace/project && git init -q -b main && git config user.email t@t && git config user.name t
cat > veyra.yaml <<'Y'
procedures:
  - name: cajita
    entry: mis_filtros.uno:cajita
    doc: Caja fija
    params: {lado: 40, score: 0.8}
Y
touch src/mis_filtros/__init__.py
cat > src/mis_filtros/uno.py <<'P'
def cajita(img, lado=10, score=0.8):
    return [{"bbox": [20, 20, lado, lado], "score": score}]
P
git add -A && git commit -qm uno && git tag v1
""",
    )  # fmt: skip
    check("repositorio del proyecto preparado", r.returncode == 0, r.stderr)

    def job(spec, **kw):
        flow = {
            "slug": SLUG,
            "spec": spec,
            "models": {"3": {"path": str(tmp / "m" / "best.pt"), "classes": ["a", "b"]}},
        }
        return {
            "id": 1,
            "payload": {"source": str(tmp / "v.mp4"), "flow": flow, "record_dir": str(tmp / "rec"), "imgsz": 320, **kw},
        }

    def start(spec, **kw):
        global container
        token, port = secrets.token_hex(8), live._pick_port(*S.live_ports)
        container = f"veyra-live-e2e-{token[:6]}"
        cmd = live.build_flow_cmd(job(spec, **kw), S, port, token, container)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines, ready = [], threading.Event()

        def pump():
            for ln in p.stdout:
                lines.append(ln)
                if ln.startswith("LISTO"):
                    ready.set()

        threading.Thread(target=pump, daemon=True).start()
        t0 = time.time()
        while not ready.is_set() and p.poll() is None and time.time() - t0 < 240:
            ready.wait(0.5)
        time.sleep(0.5)
        ok = ready.is_set()
        return p, port, token, ok, "".join(lines)[-800:]

    def stop(p):
        sh("docker", "rm", "-f", container)
        p.wait(20)

    spec = {
        "op": "or",
        "iou": 0.3,
        "inputs": [{"source": "model:3", "min_score": 0.25}, {"source": "procedure:cajita", "ref": "v1"}],
    }
    p, port, token, ok, log = start(spec)
    check("la sesión con flujo (modelo + procedimiento) arranca en el contenedor", ok, log)
    if ok:
        time.sleep(3)
        st = json.load(get(port, token, "/stats"))
        check(
            "procesa fotogramas y reporta las clases unificadas (modelo + «detección»)",
            st["frames"] > 0 and st["classes"] == ["a", "b", "detección"],
            st,
        )
        check(
            "las detecciones del procedimiento aparecen en el conteo",
            "detección" in st["totals"] or st["active"] >= 1,
            st["totals"],
        )
        fr = get(port, token, "/frame.jpg")
        dets = json.loads(fr.headers["X-Detections"])
        check(
            "el fotograma trae la caja del procedimiento (clase «detección»)",
            fr.read()[:2] == b"\xff\xd8" and any(d[:4] == [20, 20, 40, 40] and d[5] == 2 for d in dets),
            dets,
        )
        check("el umbral de la sesión no filtra el flujo", st["threshold"] == 0.0, st["threshold"])
    stop(p)
    cont = sh("docker", "container", "inspect", container)
    check("al detener la sesión el contenedor desaparece", cont.returncode != 0)

    p, port, token, ok, log = start({"source": "procedure:inexistente"})
    time.sleep(1)
    check(
        "un procedimiento inexistente hace fallar el arranque con un mensaje claro",
        not ok and "inexistente" in log,
        log,
    )
    stop(p)

    ps = sh("docker", "ps", "-aq", "--filter", "name=veyra-live-e2e")
    check("no quedan contenedores de la prueba", not ps.stdout.strip(), ps.stdout)
    print(f"\nRESUMEN: {sum(res.values())}/{len(res)} OK")
finally:
    sh("docker", "rm", "-f", container or "x")
    sh("docker", "volume", "rm", "-f", f"veyra-ws-{SLUG}")
    sh("docker", "volume", "rm", "-f", DVOL)
sys.exit(0 if res and all(res.values()) else 1)
