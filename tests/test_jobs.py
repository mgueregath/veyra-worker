import json
import time
import urllib.request

import cv2
import numpy as np

from veyra_worker import paths


def train_payload(data, **kw):
    (data / "ds" / "images" / "train").mkdir(parents=True, exist_ok=True)
    (data / "ds" / "data.yaml").write_text("x")
    return {"data": str(data / "ds" / "data.yaml"), "weights": "yolov8n.pt", "epochs": 3, "imgsz": 320, "batch": 2,
            "project": str(data / "runs" / "1" / "ul"), "name": "train", "seed": 0, "out": str(data / "models" / "m" / "best.pt"), **kw}  # fmt: skip


def test_registers_with_capabilities(env):
    assert env.wait(lambda: env.fake.registered is not None)
    r = env.fake.registered
    assert r["name"] == "test" and sorted(r["kinds"]) == ["live", "predict", "train"] and "gpus" in r["capabilities"]
    assert env.wait(lambda: env.fake.heartbeats)  # y sigue avisando que vive


def test_training_reports_progress_and_result(env):
    jid = env.fake.submit("train", train_payload(env.data))
    assert env.wait(lambda: jid in env.fake.finished)
    f = env.fake.finished[jid]
    assert f["status"] == "done" and f["result"]["weights"].endswith("best.pt") and f["result"]["map50"] == 0.5
    assert (env.data / "models" / "m" / "best.pt").exists()
    prog = [p["progress"] for p in env.fake.progress.get(jid, []) if "progress" in p]
    assert prog and prog[-1]["epoch"] <= 3
    assert "cargando datos" in f["log"]


def test_training_failure_is_reported_with_log(env):
    jid = env.fake.submit("train", train_payload(env.data, epochs=7))
    assert env.wait(lambda: jid in env.fake.finished)
    f = env.fake.finished[jid]
    assert f["status"] == "failed" and "código 3" in f["error"] and "Traceback" in f["log"]


def test_cancel_from_heartbeat_stops_the_process(env):
    jid = env.fake.submit("train", train_payload(env.data, epochs=9))  # el simulado tarda un minuto por época
    assert env.wait(lambda: jid in env.fake.progress or any(h["running_jobs"] for h in env.fake.heartbeats))
    time.sleep(0.5)
    env.fake.cancel.add(jid)
    assert env.wait(lambda: jid in env.fake.finished, 20)
    assert env.fake.finished[jid]["status"] == "cancelled"
    assert not (env.data / "models" / "m" / "best.pt").exists()


def test_paths_outside_the_shared_folder_are_refused(env):
    jid = env.fake.submit("train", train_payload(env.data, out="/tmp/fuera/best.pt"))
    assert env.wait(lambda: jid in env.fake.finished)
    f = env.fake.finished[jid]
    assert f["status"] == "failed" and "fuera de la carpeta compartida" in f["error"]
    jid2 = env.fake.submit(
        "predict", {"weights": "yolov8n.pt", "images": ["/etc/passwd"], "out": str(env.data / "o.json")}
    )
    assert env.wait(lambda: jid2 in env.fake.finished)
    assert env.fake.finished[jid2]["status"] == "failed"


def test_unknown_job_kind_fails_cleanly(env):
    jid = env.fake.submit("minar", {})
    # el reclamo solo ofrece tipos que el worker conoce: aquí se fuerza el envío directo de uno desconocido
    env.w._start({"id": jid, "kind": "minar", "payload": {}})
    assert env.wait(lambda: jid in env.fake.finished)
    assert env.fake.finished[jid]["status"] == "failed" and "desconocido" in env.fake.finished[jid]["error"]


def test_predict_writes_result_in_shared_folder(env):
    imgs = []
    for i in range(2):
        p = env.data / "media" / f"f{i}.jpg"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"\xff\xd8x\xff\xd9")
        imgs.append(str(p))
    out = env.data / "runs" / "9" / "pred.json"
    jid = env.fake.submit("predict", {"weights": "yolov8n.pt", "images": imgs, "out": str(out), "imgsz": 320})
    assert env.wait(lambda: jid in env.fake.finished)
    f = env.fake.finished[jid]
    assert f["status"] == "done" and f["result"] == {"out": str(out), "images": 2}
    raw = json.loads(out.read_text())
    assert sorted(raw) == sorted(imgs) and raw[imgs[0]][0][4] == 0.9


def make_video(path, n=75):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 96))
    for i in range(n):
        w.write(np.full((96, 160, 3), (i * 3) % 255, np.uint8))
    w.release()


def test_live_session_exposes_stream_and_stops_on_cancel(env):
    vid = env.data / "videos" / "cam.mp4"
    vid.parent.mkdir()
    make_video(vid)
    jid = env.fake.submit(
        "live",
        {
            "source": str(vid),
            "detector": "fake",
            "classes": ["louse"],
            "threshold": 0.25,
            "imgsz": 320,
            "no_realtime": False,
        },
    )
    assert env.wait(lambda: any("endpoint" in p for p in env.fake.progress.get(jid, [])), 30)
    info = next(p for p in env.fake.progress[jid] if "endpoint" in p)
    assert info["endpoint"].startswith("http://127.0.0.1:19") and len(info["token"]) == 48
    url = f"{info['endpoint']}/stats?token={info['token']}"
    assert env.wait(lambda: json.load(urllib.request.urlopen(url, timeout=5)).get("frames", 0) > 0, 20)
    try:
        urllib.request.urlopen(f"{info['endpoint']}/stats?token=mala", timeout=5)
        raise AssertionError("debió rechazar el token")
    except urllib.error.HTTPError as e:
        assert e.code == 403
    assert env.w.free_kinds().count("live") == 1  # sigue habiendo cupo (max_live=2)
    env.fake.cancel.add(jid)
    assert env.wait(lambda: jid in env.fake.finished, 20)
    assert env.fake.finished[jid]["status"] == "cancelled"
    time.sleep(0.5)
    try:
        urllib.request.urlopen(url, timeout=2)
        raise AssertionError("el proceso debió detenerse")
    except OSError:
        pass


def test_live_rejects_dangerous_sources(env):
    jid = env.fake.submit("live", {"source": "file:///etc/passwd", "detector": "fake"})
    assert env.wait(lambda: jid in env.fake.finished)
    assert env.fake.finished[jid]["status"] == "failed" and "no permitida" in env.fake.finished[jid]["error"]


def test_capacity_limits_what_is_offered(env):
    env.s.max_batch = 1
    jid = env.fake.submit("train", train_payload(env.data, epochs=9))
    assert env.wait(lambda: env.w._count(False) == 1)
    assert "train" not in env.w.free_kinds() and "predict" not in env.w.free_kinds() and "live" in env.w.free_kinds()
    env.fake.cancel.add(jid)
    assert env.wait(lambda: jid in env.fake.finished, 20)


def test_path_helpers(tmp_path):
    root = tmp_path
    assert paths.source(root, "rtsp://u:p@10.0.0.9/stream") == "rtsp://u:p@10.0.0.9/stream"
    for bad in ("file:///etc/passwd", "concat:a|b", "data:x", "ftp://x/y", "rtsp://a b"):
        try:
            paths.source(root, bad)
            raise AssertionError(bad)
        except paths.PathError:
            pass
    assert paths.weights(root, (), "yolov8n.pt") == "yolov8n.pt"
    try:
        paths.weights(root, (), "../x.pt")
        raise AssertionError
    except paths.PathError:
        pass
    (tmp_path / "link").symlink_to("/etc")
    try:
        paths.inside(root, str(tmp_path / "link" / "passwd"))
        raise AssertionError("un enlace simbólico no debe sacar de la carpeta")
    except paths.PathError:
        pass
