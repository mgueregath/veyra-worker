import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "veyra_worker" / "scripts"))

from flow_core import FlowDetector, combine_and, combine_or  # noqa: E402

A = (10, 10, 20, 20, 0.9, 0)
B = (12, 11, 20, 20, 0.5, 0)
C = (200, 200, 10, 10, 0.7, -1)


def test_or_fuses_overlaps_and_keeps_others():
    out = combine_or([[A], [B, C]])
    assert len(out) == 2
    assert out[0][4] == 0.9


def test_and_requires_all_sources():
    assert len(combine_and([[A], [B]])) == 1
    assert combine_and([[A], [C]]) == []


def test_class_incompatible_not_fused():
    assert len(combine_or([[A], [(10, 10, 20, 20, 0.8, 1)]])) == 2


def test_flow_detector_with_fake_loaders():
    bundle = {
        "spec": {
            "op": "and",
            "iou": 0.3,
            "inputs": [{"source": "model:1", "min_score": 0.6}, {"source": "procedure:p"}],
        },
        "models": {"1": {"path": "/x.pt", "classes": ["piojo"]}},
    }
    det = FlowDetector(
        bundle, load_model=lambda *_: lambda f: [A, B], load_procedure=lambda lf: lambda f: [(11, 10, 20, 20, 1.0, -1)]
    )
    assert det.names == ["piojo", "detección"]
    out = det(None)
    assert len(out) == 1 and out[0][5] == 0  # B (0.5) filtrada por min_score; clase del modelo prevalece
