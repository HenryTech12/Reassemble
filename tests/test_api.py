import time

import pytest
from fastapi.testclient import TestClient

import api.main as M
from reassemble.model import save_case
from reassemble.synth.pages import render_page
from reassemble.synth.shredders import mixed_bin, strip_shred


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "SAMPLES", tmp_path / "samples")
    monkeypatch.setattr(M, "RUNS", tmp_path / "runs")
    (tmp_path / "runs").mkdir()
    page, _ = render_page("letter", 2)
    pcs, t = strip_shred(page, 12, seed=2)
    save_case(tmp_path / "samples" / "letter_12", pcs, t)
    pages = [render_page(k, 10 + i)[0] for i, k in enumerate(["memo", "invoice"])]
    pcs, t = mixed_bin(pages, 12, seed=4)
    save_case(tmp_path / "samples" / "mixed_2", pcs, t)
    with TestClient(M.app) as c:
        yield c


def wait(client, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/api/jobs/{jid}").json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.2)
    raise TimeoutError


def test_sample_job_lifecycle_and_edits(client):
    names = [s["name"] for s in client.get("/api/samples").json()]
    assert "letter_12" in names
    jid = client.post("/api/jobs/sample", json={"sample": "letter_12", "scorer": "handcrafted"}).json()["id"]
    with client.websocket_connect(f"/ws/jobs/{jid}?name=ada") as ws:
        assert ws.receive_json()["type"] == "hello"
        s = wait(client, jid)
        assert s["status"] == "done", s["error"]
        assert s["metrics"]["neighbour_content"] >= 0.9
        doc = s["documents"][0]
        order = [p for p in doc["order"] if p]
        # manual edit: swap two pieces
        order[0], order[1] = order[1], order[0]
        r = client.post(f"/api/jobs/{jid}/order", json={"doc": 0, "order": order}).json()
        assert [p for p in r["documents"][0]["order"] if p] == order
        assert r["seq"] > s["seq"]
        # lock a join and re-solve around it
        a, b = order[3], order[7]
        r = client.post(f"/api/jobs/{jid}/resolve", json={"doc": 0, "locked": [[a, b]]}).json()
        o = [p for p in r["documents"][0]["order"] if p]
        assert o[o.index(a) + 1] == b
        assert client.get(f"/api/jobs/{jid}/corrections").json()["pairs"][0]["label"] == 1
    assert client.get(f"/api/jobs/{jid}/render/0.png?heatmap=true").headers["content-type"] == "image/png"
    assert client.get(f"/api/jobs/{jid}/pieces/{order[0]}.png").status_code == 200
    g = client.get(f"/api/jobs/{jid}/replay.gif")
    assert g.status_code == 200 and g.content[:3] == b"GIF"


def test_mixed_job_and_move_piece(client):
    jid = client.post("/api/jobs/sample", json={"sample": "mixed_2", "scorer": "handcrafted"}).json()["id"]
    s = wait(client, jid)
    assert len(s["documents"]) == 2
    piece = s["documents"][0]["pieces"][0]
    r = client.post(f"/api/jobs/{jid}/move_doc", json={"piece": piece, "doc": 1}).json()
    owner = [d["doc"] for d in r["documents"] if piece in d["pieces"]]
    assert owner and r["labels"][piece] == owner[0]


def test_upload_pieces(client, tmp_path):
    page, _ = render_page("memo", 3)
    pcs, _ = strip_shred(page, 6, seed=1)
    files = []
    import io
    from PIL import Image
    for p in pcs:
        buf = io.BytesIO()
        Image.fromarray(p.image).save(buf, "PNG")
        files.append(("files", (f"{p.id}.png", buf.getvalue(), "image/png")))
    jid = client.post("/api/jobs", files=files, data={"mode": "pieces", "scorer": "handcrafted"}).json()["id"]
    s = wait(client, jid)
    assert s["status"] == "done" and len(s["pieces"]) == 6
