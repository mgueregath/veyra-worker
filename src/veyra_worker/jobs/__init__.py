"""Manejadores de trabajos: cada uno recibe (contexto, trabajo, configuración) y devuelve un Outcome."""

from dataclasses import dataclass, field


@dataclass
class Outcome:
    status: str  # done | failed | cancelled
    result: dict | None = None
    error: str = ""
    log: str = field(default="", repr=False)
