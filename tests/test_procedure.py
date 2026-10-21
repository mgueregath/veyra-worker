import pytest

from veyra_worker.config import Settings
from veyra_worker.jobs import procedure as pr
from veyra_worker.jobs.context import JobContext


class Api:
    def progress(self, *_):
        return {}


def settings(tmp_path):
    return Settings(
        api_url="http://x", token="t", data_root=tmp_path, workspaces=True, data_volume="vol", workspace_image="img:1"
    )


def job(tmp_path, **kw):
    import threading

    (tmp_path / "runs").mkdir(exist_ok=True)
    (tmp_path / "media").mkdir(exist_ok=True)
    (tmp_path / "media" / "a.jpg").write_bytes(b"x")
    p = {
        "action": "run",
        "slug": "piojos",
        "name": "manchas",
        "ref": "HEAD",
        "params": {},
        "images": [str(tmp_path / "media" / "a.jpg")],
        "out_dir": str(tmp_path / "runs" / "o1"),
        **kw,
    }
    return JobContext(Api(), 1, threading.Event()), {"id": 1, "payload": p}


@pytest.mark.parametrize("bad", [
    {"slug": "../x"}, {"slug": "Piojos"}, {"action": "borrar"}, {"name": "a b"}, {"name": "a;b"}, {"name": ""},
    {"ref": "--upload-pack=evil"}, {"ref": "-x"}, {"ref": "a b"}, {"params": [1, 2]},
    {"images": ["/etc/passwd"]}, {"out_dir": "/tmp/fuera"},
])  # fmt: skip
def test_invalid_payload_is_refused_before_touching_docker(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(pr, "_docker", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no debe llamar a docker")))
    ctx, j = job(tmp_path, **bad)
    out = pr.run(ctx, j, settings(tmp_path))
    assert out.status == "failed" and out.error


def test_list_without_a_workspace_volume_explains_instead_of_creating_one(tmp_path, monkeypatch):
    import subprocess

    calls = []
    monkeypatch.setattr(
        pr, "_docker", lambda s, *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 1, "", "no such volume")
    )
    ctx, j = job(tmp_path, action="list")
    out = pr.run(ctx, j, settings(tmp_path))
    assert out.status == "done" and out.result["procedures"] == [] and "espacio de trabajo" in out.result["error"]
    assert calls == [("volume", "inspect", "veyra-ws-piojos")]  # solo mira: `docker run -v` crearía el volumen vacío


def test_run_command_is_isolated(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(pr, "_volume_exists", lambda s, slug: True)
    monkeypatch.setattr(pr, "run_streaming", lambda cmd, on_line, cancel, env=None: seen.update(cmd=cmd) or (0, False))
    ctx, j = job(tmp_path, cpus=2, memory="4g", gpu=True)
    (tmp_path / "runs" / "o1").mkdir(parents=True)
    (tmp_path / "runs" / "o1" / "result.json").write_text('{"ref": "x", "results": {}}')
    out = pr.run(ctx, j, settings(tmp_path))
    c = seen["cmd"]
    assert out.status == "done"
    assert c[c.index("--network") + 1] == "none" and c[c.index("--user") + 1] == "1000:3000"
    assert "veyra-ws-piojos:/workspace:ro" in c and "vol:/data:ro" in c and "--privileged" not in c
    assert any(
        x.startswith("type=volume,src=vol,dst=/out,volume-subpath=runs/o1") for x in c
    )  # solo escribe en su carpeta
    assert c[c.index("--entrypoint") + 1] == "veyra-proc" and c[-1] == "/out/result.json"
