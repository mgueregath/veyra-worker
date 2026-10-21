import json

import pytest

from veyra_worker.config import Settings
from veyra_worker.jobs import live
from veyra_worker.jobs.workspace import WorkspaceError


def settings(tmp_path):
    return Settings(api_url="http://x", token="t", data_root=tmp_path, data_volume="vol", workspace_image="img:1")


def job(tmp_path, **flow):
    (tmp_path / "m").mkdir(exist_ok=True)
    (tmp_path / "m" / "best.pt").write_bytes(b"x")
    (tmp_path / "v.mp4").write_bytes(b"x")
    f = {
        "slug": "piojos",
        "spec": {
            "op": "or",
            "iou": 0.3,
            "inputs": [{"source": "model:1", "min_score": 0.3}, {"source": "procedure:manchas", "ref": "v1"}],
        },
        "models": {"1": {"path": str(tmp_path / "m" / "best.pt"), "classes": ["piojo"]}},
    }
    f.update(flow)
    return {"payload": {"source": str(tmp_path / "v.mp4"), "flow": f, "record_dir": str(tmp_path / "rec")}}


def test_command_isolated_and_paths_mapped(tmp_path):
    cmd = live.build_flow_cmd(job(tmp_path), settings(tmp_path), 9000, "tok", "veyra-live-x")
    assert cmd[:3] == ["docker", "run", "--rm"] and "--network" not in cmd
    assert "veyra-ws-piojos:/workspace:ro" in cmd and "vol:/data:ro" in cmd and "9000:9000" in cmd
    assert "/data/v.mp4" in cmd and "--flow" in cmd and "/rec" in cmd
    env = json.loads(next(c for c in cmd if c.startswith("VEYRA_FLOW="))[11:])
    assert env["models"]["1"]["path"] == "/data/m/best.pt"
    assert str(tmp_path) not in " ".join(cmd)


@pytest.mark.parametrize(
    "mut",
    [
        {"slug": "../x"},
        {"spec": {"source": "baseline:appearance"}},
        {"spec": {"source": "procedure:a b"}},
        {"spec": {"source": "procedure:ok", "ref": "-x"}},
        {"spec": {"source": "model:9"}},
        {"models": {"1": {"path": "/etc/passwd", "classes": []}}},
    ],
)
def test_rejects_bad_flows(tmp_path, mut):
    with pytest.raises((WorkspaceError, ValueError)):
        live.build_flow_cmd(job(tmp_path, **mut), settings(tmp_path), 9000, "tok", "n")
