import shutil
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
from fakeapi import FakeApi

from veyra_worker.api import ApiClient
from veyra_worker.config import Settings
from veyra_worker.runner import Worker

HERE = Path(__file__).parent
REAL_SCRIPTS = Path(__file__).resolve().parents[1] / "src" / "veyra_worker" / "scripts"


@pytest.fixture()
def env(tmp_path):
    """Worker real contra la API simulada, con scripts simulados para entrenar y predecir (sin GPU)."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(HERE / "fake_train.py", scripts / "train_yolo.py")
    shutil.copy(HERE / "fake_predict.py", scripts / "predict_yolo.py")
    shutil.copy(REAL_SCRIPTS / "live_yolo.py", scripts / "live_yolo.py")
    shutil.copy(REAL_SCRIPTS / "live_core.py", scripts / "live_core.py")
    data = tmp_path / "data"
    data.mkdir()
    s = Settings(api_url="http://api.test", token="secreto", name="test", data_root=data, python=sys.executable,
                 scripts_dir=scripts, advertise_host="127.0.0.1", live_ports=(19100, 19160), poll_wait=0.2,
                 heartbeat_every=0.1, live_start_timeout=30)  # fmt: skip
    fake = FakeApi()
    api = ApiClient(s.api_url, s.token, transport=httpx.MockTransport(fake))
    w = Worker(s, api)
    t = threading.Thread(target=w.serve, daemon=True)
    t.start()

    def wait(cond, timeout=20.0):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return True
            time.sleep(0.05)
        return False

    yield type("Env", (), {"s": s, "fake": fake, "w": w, "wait": staticmethod(wait), "data": data})
    w.shutdown(5)
    t.join(5)
    api.close()
