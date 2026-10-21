"""Las rutas de los trabajos las manda la API, pero el worker solo toca lo que esté dentro de la carpeta compartida."""

import re
from pathlib import Path

PRETRAINED_NAME = re.compile(r"^yolov8[nsmlx]\.pt$")  # nombre suelto: Ultralytics lo descarga
URL_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://")


class PathError(ValueError):
    pass


def inside(root: Path, p: str) -> Path:
    """La ruta resuelta debe quedar dentro de `root` (se siguen los enlaces simbólicos)."""
    root = root.resolve()
    full = Path(p).resolve()
    if full != root and root not in full.parents:
        raise PathError(f"ruta fuera de la carpeta compartida: {p}")
    return full


def weights(root: Path, extra: tuple[str, ...], w: str) -> str:
    if PRETRAINED_NAME.match(w):
        for d in extra:
            if (Path(d) / w).is_file():
                return str(Path(d) / w)
        return w
    try:
        return str(inside(root, w))
    except PathError:
        for d in extra:
            try:
                return str(inside(Path(d), w))
            except PathError:
                continue
        raise


def source(root: Path, s: str) -> str:
    """Fuente de video: URL de cámara (sin esquemas como file: que FFmpeg también lee) o archivo compartido."""
    if "://" in s or s.lower().startswith(("file:", "concat:", "data:")):
        if not s.lower().startswith(URL_SCHEMES) or any(c in s for c in " \t\r\n\x00"):
            raise PathError("fuente no permitida (use rtsp://, rtsps://, http://, https:// o un archivo compartido)")
        return s
    return str(inside(root, s))
