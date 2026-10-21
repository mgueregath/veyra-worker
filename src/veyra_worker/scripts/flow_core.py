"""Flujos de detección dentro del proceso en vivo: varias fuentes por fotograma combinadas con OR (unión) / AND (coincidencia).

Es la misma lógica que `app/combine.py` y `app/flows.py` de la API de Veyra (se mantienen en paralelo y se prueban con los
mismos casos), pero sobre detecciones en el formato del proceso en vivo: (x, y, w, h, puntaje, índice_de_clase), con -1 =
sin clase. Fuentes: `model:<id>` (YOLO, un modelo por hoja) y `procedure:<nombre>` (función del repositorio del proyecto,
en el commit indicado). Solo se importa torch/ultralytics si el flujo tiene modelos.
"""

import json
from collections.abc import Callable

Det = tuple[float, float, float, float, float, int]
SCORE: dict[str, Callable] = {"max": max, "min": min, "mean": lambda xs: sum(xs) / len(xs)}
BLANK = "detección"  # clase de las detecciones que no traen clase (procedimientos sin label)


def iou(a, b) -> float:
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def compatible(a: int, b: int) -> bool:
    return a < 0 or b < 0 or a == b


def _merge(group: list[Det], score: Callable) -> Det:
    w = [max(d[4], 1e-6) for d in group]
    tot = sum(w)
    box = [sum(d[i] * wi for d, wi in zip(group, w, strict=True)) / tot for i in range(4)]
    cls = next((d[5] for d in sorted(group, key=lambda d: -d[4]) if d[5] >= 0), -1)
    return (
        round(box[0]),
        round(box[1]),
        max(1, round(box[2])),
        max(1, round(box[3])),
        float(score([d[4] for d in group])),
        cls,
    )


def combine_or(inputs: list[list[Det]], thr: float = 0.3, score: str = "max") -> list[Det]:
    groups: list[list[Det]] = []
    for d in sorted((d for lst in inputs for d in lst), key=lambda d: -d[4]):
        for g in groups:
            if compatible(g[0][5], d[5]) and iou(g[0], d) >= thr:
                g.append(d)
                break
        else:
            groups.append([d])
    return sorted((_merge(g, SCORE[score]) for g in groups), key=lambda d: -d[4])


def combine_and(inputs: list[list[Det]], thr: float = 0.3, score: str = "mean") -> list[Det]:
    if not inputs:
        return []
    if len(inputs) == 1:
        return list(inputs[0])
    used: list[set[int]] = [set() for _ in inputs]
    out = []
    for i, anchor in sorted(((i, d) for i, lst in enumerate(inputs) for d in lst), key=lambda x: -x[1][4]):
        if id(anchor) in used[i]:
            continue
        group, picks = [anchor], [(i, anchor)]
        for j, lst in enumerate(inputs):
            if j == i:
                continue
            cands = [q for q in lst if id(q) not in used[j] and compatible(anchor[5], q[5]) and iou(anchor, q) >= thr]
            if not cands:
                break
            q = max(cands, key=lambda q: iou(anchor, q))
            group.append(q)
            picks.append((j, q))
        else:
            for j, q in picks:
                used[j].add(id(q))
            out.append(_merge(group, SCORE[score]))
    return sorted(out, key=lambda d: -d[4])


def combine(op: str, inputs: list[list[Det]], thr: float = 0.3, score: str | None = None) -> list[Det]:
    if op == "or":
        return combine_or(inputs, thr, score or "max")
    if op == "and":
        return combine_and(inputs, thr, score or "mean")
    raise ValueError(f"operación desconocida: {op}")


def leaves(node: dict) -> list[dict]:
    return [node] if "op" not in node else [x for c in node["inputs"] for x in leaves(c)]


class FlowDetector:
    """Detector de un flujo: se llama con el fotograma y devuelve las detecciones ya combinadas."""

    def __init__(self, bundle: dict, imgsz: int = 1280, device: str = "0", load_model=None, load_procedure=None):
        self.spec = bundle["spec"]
        self.models_info = bundle.get("models", {})
        self.leaves = leaves(self.spec)
        names: list[str] = []
        for lf in self.leaves:
            if lf["source"].startswith("model:"):
                for c in self.models_info[lf["source"][6:]]["classes"]:
                    if c not in names:
                        names.append(c)
        self.has_blank = any(lf["source"].startswith("procedure:") for lf in self.leaves)
        if self.has_blank or not names:
            names.append(BLANK)
        self.names = names
        self.blank = names.index(BLANK) if BLANK in names else -1
        self._fns: dict[str, Callable] = {}
        for lf in self.leaves:
            key = _key(lf)
            if key in self._fns:
                continue
            src = lf["source"]
            if src.startswith("model:"):
                self._fns[key] = (load_model or self._load_model)(src[6:], imgsz, device)
            elif src.startswith("procedure:"):
                self._fns[key] = (load_procedure or self._load_procedure)(lf)
            else:
                raise ValueError(f"fuente no admitida en vivo: {src}")

    def _load_model(self, mid: str, imgsz: int, device: str):
        from live_core import YoloDetector

        info = self.models_info[mid]
        det = YoloDetector(info["path"], imgsz, device)
        local = list(info["classes"])
        remap = {i: self.names.index(c) if c in self.names else -1 for i, c in enumerate(local)}
        return lambda frame: [(x, y, w, h, s, remap.get(k, -1)) for x, y, w, h, s, k in det(frame)]

    def _load_procedure(self, leaf: dict):
        import procs  # módulo hermano (live_yolo agrega esta carpeta al path)

        fn, defaults, _sha = procs.load(leaf["source"][10:], leaf.get("ref") or "HEAD")
        params = {**defaults, **(leaf.get("params") or {})}

        def run(frame) -> list[Det]:
            out = []
            for d in fn(frame, **params) or []:
                x, y, w, h = (float(v) for v in d["bbox"])
                if w <= 0 or h <= 0:
                    continue
                label = d.get("label") or ""
                out.append(
                    (
                        round(x),
                        round(y),
                        round(w),
                        round(h),
                        float(d.get("score", 1.0)),
                        self.names.index(label) if label in self.names else -1,
                    )
                )
            return out

        return run

    def _eval(self, node: dict, frame, cache: dict) -> list[Det]:
        if "op" not in node:
            key = _key(node)
            if key not in cache:
                cache[key] = self._fns[key](frame)
            return [d for d in cache[key] if d[4] >= float(node.get("min_score") or 0.0)]
        parts = [self._eval(c, frame, cache) for c in node["inputs"]]
        return combine(node["op"], parts, float(node.get("iou") or 0.3), node.get("score"))

    def __call__(self, frame) -> list[Det]:
        out = self._eval(self.spec, frame, {})
        return [(x, y, w, h, s, k if k >= 0 else self.blank) for x, y, w, h, s, k in out]


def _key(leaf: dict) -> str:
    return json.dumps({"s": leaf["source"], "r": leaf.get("ref"), "p": leaf.get("params") or {}}, sort_keys=True)
