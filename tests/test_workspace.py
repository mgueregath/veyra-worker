import pytest

from veyra_worker.config import Settings
from veyra_worker.jobs import workspace as ws

KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAdcTYJBw5A2ja8cNd1BUztHAk5zaMGLUwWZO4IU7CRD usuario@portatil"


def settings(tmp_path):
    return Settings(api_url="http://x", token="t", data_root=tmp_path, workspaces=True, advertise_host="10.0.0.4",
                    data_volume="veyra-worker_veyra_data", workspace_image="img:1")  # fmt: skip


def payload(**kw):
    return {"slug": "piojos", "port": 9101, "keys": [KEY], **kw}


def test_validate_accepts_a_normal_request(tmp_path):
    v = ws.validate(payload(idle_minutes=30, cpus=4, memory="8g"), settings(tmp_path))
    assert v["slug"] == "piojos" and v["idle"] == 30 and v["cpus"] == 4 and v["memory"] == "8g" and v["gpu"] is True


@pytest.mark.parametrize("bad", [
    {"slug": "../etc"}, {"slug": "Piojos"}, {"slug": "a b"}, {"slug": ""}, {"slug": "x" * 60},
    {"port": 22}, {"port": 9200}, {"port": 8899},
    {"keys": []}, {"keys": ["no es una llave"]}, {"keys": [KEY + "\nssh-rsa AAAA evil"]}, {"keys": ["ssh-ed25519 AAAA$(rm -rf)"]},
    {"memory": "999999g"}, {"memory": "12gb;"}, {"cpus": 500},
])  # fmt: skip
def test_validate_rejects_dangerous_or_malformed_input(tmp_path, bad):
    with pytest.raises((ws.WorkspaceError, ValueError)):
        ws.validate(payload(**bad), settings(tmp_path))


def test_run_command_is_built_from_validated_fields_only(tmp_path):
    s = settings(tmp_path)
    v = ws.validate(payload(cpus=2, memory="4g", gpu=True), s)
    cmd = ws.run_args(v, s, "10.0.0.4")
    assert cmd[:5] == ["docker", "run", "-d", "--name", "veyra-ws-piojos"]
    assert (
        "10.0.0.4:9101:22" in cmd and "veyra-ws-piojos:/workspace" in cmd and "veyra-worker_veyra_data:/data:ro" in cmd
    )
    assert cmd[cmd.index("--device") + 1] == "nvidia.com/gpu=all" and cmd[-1] == "img:1"
    assert "--privileged" not in cmd and not any(c.startswith("--cap-add") for c in cmd) and "--network" not in cmd
    assert cmd[cmd.index("--cpus") + 1] == "2.0" and cmd[cmd.index("--memory") + 1] == "4g"
    no_gpu = ws.run_args(ws.validate(payload(gpu=False), s), s, "10.0.0.4")
    assert "--device" not in no_gpu


def test_runner_offers_workspaces_only_when_docker_is_usable(tmp_path, monkeypatch):
    import httpx

    from veyra_worker.api import ApiClient
    from veyra_worker.runner import Worker

    s = settings(tmp_path)
    api = ApiClient(s.api_url, s.token, transport=httpx.MockTransport(lambda r: httpx.Response(204)))
    monkeypatch.setattr(ws, "available", lambda _s: False)
    assert "workspace" not in Worker(s, api).handlers
    monkeypatch.setattr(ws, "available", lambda _s: True)
    w = Worker(s, api)
    assert "workspace" in w.handlers and "workspace" in w.free_kinds()
    s.workspaces = False
    assert "workspace" not in Worker(s, api).handlers
