"""Lo que este equipo puede ofrecer (se informa a la API al registrarse)."""

import importlib.metadata as md
import shutil
import subprocess

from . import __version__
from .config import Settings


def gpus() -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return []
    res = []
    for line in out.strip().splitlines():
        name, mem, drv = (x.strip() for x in line.split(","))
        res.append({"name": name, "memory_mb": int(mem), "driver": drv})
    return res


def version_of(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def capabilities(s: Settings) -> dict:
    return {
        "version": __version__,
        "gpus": gpus(),
        "torch": version_of("torch"),
        "ultralytics": version_of("ultralytics"),
        "data_root": str(s.data_root),
        "data_root_exists": s.data_root.is_dir(),
        "device": s.device,
    }
