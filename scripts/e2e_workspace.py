#!/usr/bin/env python3
"""Prueba real con Docker del espacio de trabajo: SSH solo con llave, GPU, datos en /data, dependencias, git y apagado.

python scripts/e2e_workspace.py [imagen]        (requiere la imagen veyra-workspace construida y el volumen NFS del worker)
"""

import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from veyra_worker.config import Settings  # noqa: E402
from veyra_worker.jobs import workspace as ws  # noqa: E402
from veyra_worker.jobs.context import JobContext  # noqa: E402

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "docker.labs.hapticus.cl/veyra-workspace:0.3.0"
SLUG, PORT = "e2etest", 9199
tmp = Path(tempfile.mkdtemp(prefix="wsE2E_"))
res = {}


def check(name, ok, extra=""):
    print(("OK    " if ok else "FALLA ") + name + "  " + str(extra)[:200])
    res[name] = bool(ok)


def sh(*cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


for n in ("good", "bad"):
    sh("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp / n))
pub = (tmp / "good.pub").read_text().strip()
S = Settings(api_url="http://x", token="t", data_root=tmp, workspaces=True, advertise_host="127.0.0.1", workspace_image=IMAGE,
             data_volume="veyra-worker_veyra_data", workspace_ports=(9190, 9199))  # fmt: skip


class Api:
    def __init__(self):
        self.p = []

    def progress(self, jid, body):
        self.p.append(body)
        return {}


def ssh(cmd, key="good", port=PORT, timeout=120):
    return sh("ssh", "-i", str(tmp / key), "-p", str(port), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
              "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "LogLevel=ERROR", "veyra@127.0.0.1", cmd, timeout=timeout)  # fmt: skip


def start(idle=60):
    api = Api()
    cancel = threading.Event()
    ctx = JobContext(api, 1, cancel)
    out = {}
    t = threading.Thread(target=lambda: out.update(o=ws.run(ctx, {"id": 1, "payload": {"slug": SLUG, "port": PORT, "keys": [pub], "idle_minutes": idle, "cpus": 4, "memory": "8g", "gpu": True}}, S)), daemon=True)  # fmt: skip
    t.start()
    t0 = time.time()
    while time.time() - t0 < 200 and not any("port" in p for p in api.p) and t.is_alive():
        time.sleep(0.5)
    return api, cancel, t, out


def stop(cancel, t, out):
    cancel.set()
    t.join(90)
    return out.get("o")


try:
    sh("docker", "rm", "-f", ws.container_name(SLUG))
    sh("docker", "volume", "rm", "-f", f"veyra-ws-{SLUG}")
    api, cancel, t, out = start()
    info = next((p for p in api.p if "port" in p), None)
    check(
        "el espacio arranca y avisa dónde conectarse", info and info["port"] == PORT and info["user"] == "veyra", info
    )
    r = ssh("whoami; pwd; git -C /workspace/project log --oneline | head -2; ls /workspace/project")
    check(
        "se entra por SSH con la llave autorizada y el proyecto ya es un repositorio git",
        r.returncode == 0 and "veyra" in r.stdout and "Espacio de trabajo" in r.stdout and "README.md" in r.stdout,
        r.stdout + r.stderr,
    )
    check("una llave no autorizada es rechazada", ssh("true", key="bad").returncode != 0)
    check("la contraseña no es un método de acceso", "password" not in sh("ssh", "-p", str(PORT), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes", "-o", "PreferredAuthentications=password", "veyra@127.0.0.1", "true").stdout.lower())  # fmt: skip
    r = ssh("python -c 'import torch,sys;print(\"cuda\",torch.cuda.is_available(), torch.cuda.get_device_name(0))'")
    check("ve la GPU con PyTorch", "cuda True" in r.stdout, r.stdout + r.stderr[-200:])
    r = ssh("ls /data | tr '\\n' ' '; ls /data/media | head -2; touch /data/x 2>&1 | head -1")
    check(
        "lee los datos del NFS en /data y no puede escribir en ellos",
        "media" in r.stdout and "videos" in r.stdout and ("Read-only" in r.stdout or "denied" in r.stdout),
        r.stdout,
    )
    r = ssh("pip install -q cowsay 2>&1 | tail -1; python -c 'import cowsay;print(\"cowsay-ok\")'; which python")
    check(
        "se pueden instalar dependencias de Python (quedan en el entorno virtual del proyecto)",
        "cowsay-ok" in r.stdout and "/workspace/.venv/bin/python" in r.stdout,
        r.stdout + r.stderr[-200:],
    )
    r = ssh("veyra-apt install tree >/dev/null 2>&1; tree --version | head -1; cat /workspace/project/packages.apt")
    check(
        "se pueden instalar paquetes del sistema y quedan anotados en packages.apt",
        "tree" in r.stdout.lower() and r.stdout.strip().endswith("tree"),
        r.stdout + r.stderr[-200:],
    )
    r = ssh(
        "cd /workspace/project && echo 'print(1)' > proceso.py && git add proceso.py && git commit -q -m 'primer proceso' && echo 'sin-commit' > borrador.txt && git log --oneline | head -1"
    )
    check("se puede versionar con git", "primer proceso" in r.stdout, r.stdout + r.stderr)
    fp1 = sh("ssh-keyscan", "-p", str(PORT), "127.0.0.1").stdout
    r = ssh("true", timeout=30)
    o = stop(cancel, t, out)
    check(
        "al detenerlo (cancelar) se apaga el contenedor y se guarda una copia automática",
        o and o.status == "cancelled" and "autosave" in str(o.result),
        o and o.result,
    )
    check(
        "el contenedor ya no existe pero el volumen sí",
        sh("docker", "container", "inspect", ws.container_name(SLUG)).returncode != 0
        and sh("docker", "volume", "inspect", f"veyra-ws-{SLUG}").returncode == 0,
    )
    r = sh(
        "docker",
        "run",
        "--rm",
        "-v",
        f"veyra-ws-{SLUG}:/workspace",
        "--entrypoint",
        "git",
        IMAGE,
        "-c",
        "safe.directory=*",
        "-C",
        "/workspace/project",
        "for-each-ref",
        "refs/veyra/autosave",
    )
    check(
        "la copia automática incluye el archivo sin commitear", "refs/veyra/autosave/" in r.stdout, r.stdout + r.stderr
    )
    # segundo arranque: todo persiste
    api, cancel, t, out = start(idle=0.2)
    r = ssh(
        "python -c 'import cowsay;print(\"persiste\")'; cat /workspace/project/proceso.py; tree --version | head -1; ls /workspace/project/borrador.txt"
    )
    check(
        "al volver a encender persisten dependencias, código y paquetes del sistema",
        "persiste" in r.stdout and "print(1)" in r.stdout and "tree" in r.stdout.lower() and "borrador.txt" in r.stdout,
        r.stdout + r.stderr[-200:],
    )
    fp2 = sh("ssh-keyscan", "-p", str(PORT), "127.0.0.1").stdout
    check("la huella SSH del espacio no cambia entre arranques", fp1 and fp1.split()[-1] == fp2.split()[-1])
    # apagado por inactividad (idle 0,2 min = 12 s) sin conexiones
    t0 = time.time()
    t.join(150)
    o = out.get("o")
    check(
        "sin conexiones ni carga se apaga solo",
        o and o.status == "done" and o.result["reason"] == "idle" and time.time() - t0 < 150,
        o and (o.status, o.result),
    )
    check(
        "el contenedor desapareció tras el apagado por inactividad",
        sh("docker", "container", "inspect", ws.container_name(SLUG)).returncode != 0,
    )
    print(f"\nRESUMEN: {sum(res.values())}/{len(res)} OK")
finally:
    sh("docker", "rm", "-f", ws.container_name(SLUG))
    sh("docker", "volume", "rm", "-f", f"veyra-ws-{SLUG}")
sys.exit(0 if res and all(res.values()) else 1)
