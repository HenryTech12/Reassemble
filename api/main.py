"""Reassemble HTTP + WebSocket API.

Run: ``uvicorn api.main:app --reload`` (or ``make api``).

Design notes
------------
* A **job** owns a folder under ``runs/<id>/`` (uploaded pieces, renders) plus in-memory
  state: the prepared pieces, cached cost matrices and the current reconstruction.
* Solving runs in a worker thread.  Progress events are pushed to every WebSocket client
  of the job (that's what drives the live animation).
* **Collaboration**: every edit goes through the server, which stamps it with a
  monotonically increasing ``seq`` and broadcasts the new state.  Last write wins, with
  server-side ordering, which is simple and good enough for a handful of people working on
  one page.  Cursor positions and presence are relayed as-is.
"""

from __future__ import annotations

import asyncio
import io
import json
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel

from reassemble.model import Piece, Reconstruction, load_case, save_rgba
from reassemble.pipeline import (Prepared, SolveOptions, build_reconstruction, cost_matrices, detect_shape,
                                 prepare, solve_grid_pieces, solve_mixed, solve_prepared, solve_torn_pieces,
                                 subset)

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(ROOT / "runs")
SAMPLES = Path(ROOT / "samples")
RUNS.mkdir(exist_ok=True)

LOOP: asyncio.AbstractEventLoop | None = None


@asynccontextmanager
async def lifespan(_app):
    global LOOP
    LOOP = asyncio.get_running_loop()  # worker threads post WebSocket messages onto it
    yield


app = FastAPI(title="Reassemble API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- job state


@dataclass
class Job:
    id: str
    folder: Path
    status: str = "queued"           # queued | running | done | error
    shape: str = "strip"
    error: str | None = None
    progress: list[dict] = field(default_factory=list)
    prep: Prepared | None = None
    pieces: dict[str, Piece] = field(default_factory=dict)
    labels: dict[str, int] = field(default_factory=dict)
    documents: list[dict] = field(default_factory=list)   # one reconstruction dict per document
    constraints: dict[int, dict] = field(default_factory=dict)  # doc -> {locked, forbidden}
    cost_cache: dict[int, tuple[list[str], np.ndarray]] = field(default_factory=dict)
    truth: dict | None = None
    seq: int = 0
    corrections: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    sockets: dict[str, tuple[WebSocket, dict]] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict:
        return {"id": self.id, "status": self.status, "shape": self.shape, "error": self.error,
                "seq": self.seq, "progress": self.progress[-20:], "documents": self.documents,
                "labels": self.labels, "constraints": {str(k): v for k, v in self.constraints.items()},
                "pieces": {pid: {"w": p.width, "h": p.height} for pid, p in self.pieces.items()},
                "online": [info for _, info in self.sockets.values()], "extra": self.extra,
                "metrics": self.extra.get("metrics")}


JOBS: dict[str, Job] = {}


async def _broadcast(job: Job, msg: dict, exclude: str | None = None) -> None:
    dead = []
    for cid, (ws, _) in list(job.sockets.items()):
        if cid == exclude:
            continue
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(cid)
    for cid in dead:
        job.sockets.pop(cid, None)


def broadcast_threadsafe(job: Job, msg: dict) -> None:
    if LOOP is not None and job.sockets:
        asyncio.run_coroutine_threadsafe(_broadcast(job, msg), LOOP)


def get_job(job_id: str) -> Job:
    if job_id not in JOBS:
        raise HTTPException(404, "no such job")
    return JOBS[job_id]


# --------------------------------------------------------------------------- solving


def _emit(job: Job, event: dict) -> None:
    event = dict(event, t=round(time.time() - job.created, 2))
    job.progress.append(event)
    broadcast_threadsafe(job, {"type": "progress", "event": event})


def _evaluate(job: Job) -> None:
    """If the job came from a synthetic sample, attach accuracy metrics."""
    t = job.truth
    if not t:
        return
    from reassemble.metrics import evaluate_mixed, evaluate_strips
    blank = job.prep.blank if job.prep else set()
    if t.get("kind") == "mixed":
        job.extra["metrics"] = evaluate_mixed(job.documents, t, blank)
    elif t.get("kind") == "strip" and job.documents:
        m = evaluate_strips(job.documents[0]["order"], t, blank)
        m.pop("gaps", None)
        job.extra["metrics"] = m


def _finish(job: Job) -> None:
    job.status = "done"
    job.seq += 1
    _evaluate(job)
    broadcast_threadsafe(job, {"type": "state", "state": job.public()})


def run_solve(job: Job, opts: SolveOptions) -> None:
    try:
        job.status = "running"
        _emit(job, {"stage": "loading"})
        pieces, truth = load_case(job.folder)
        job.truth = truth
        if not pieces:
            raise ValueError("no pieces found")
        kind = (truth or {}).get("kind")
        job.shape = {"crosscut": "grid", "torn": "torn"}.get(kind) or detect_shape(pieces)
        if job.shape != "strip":
            rec = solve_grid_pieces(pieces, truth.get("rows") if truth else None,
                                    truth.get("cols") if truth else None) if job.shape == "grid" \
                else solve_torn_pieces(pieces)
            job.pieces = {p.id: p for p in pieces}
            d = rec.to_dict()
            d["doc"], d["pieces"] = 0, [p.id for p in pieces]
            job.documents = [d]
            _finish(job)
            return
        _emit(job, {"stage": "features", "n": len(pieces)})
        job.prep = prepare(pieces)
        job.pieces = {p.id: p for p in job.prep.pieces}
        rec = solve_mixed(job.prep, opts, progress=lambda e: _emit(job, e))
        job.documents = rec.documents or [dict(rec.to_dict(), doc=0, pieces=job.prep.ids)]
        job.labels = rec.extra.get("labels", {})
        job.extra["scorer"] = (job.documents[0].get("extra") or {}).get("scorer")
        _finish(job)
    except Exception as e:  # surface errors to the UI instead of dying silently
        job.status, job.error = "error", f"{type(e).__name__}: {e}"
        broadcast_threadsafe(job, {"type": "state", "state": job.public()})


def start(job: Job, opts: SolveOptions | None = None) -> None:
    JOBS[job.id] = job
    threading.Thread(target=run_solve, args=(job, opts or SolveOptions()), daemon=True).start()


def _doc_cost(job: Job, doc: int) -> tuple[Prepared, np.ndarray]:
    """Prepared subset + raw cost matrix for one document (cached between edits)."""
    ids = job.documents[doc]["pieces"]
    idx = [job.prep.ids.index(i) for i in ids]
    sub = subset(job.prep, idx)
    cached = job.cost_cache.get(doc)
    if cached is None or cached[0] != ids:
        D = cost_matrices(sub, SolveOptions())["raw"]
        job.cost_cache[doc] = (ids, D)
    return sub, job.cost_cache[doc][1]


# --------------------------------------------------------------------------- REST


class SampleRequest(BaseModel):
    sample: str
    scorer: str = "auto"
    ocr: bool = False


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "jobs": len(JOBS)}


@app.get("/api/mat.pdf")
def mat_pdf() -> FileResponse:
    """Printable capture mat (generated on first request)."""
    from reassemble.capture.mat import save_mat_pdf
    path = ROOT / "assets" / "capture_mat.pdf"
    if not path.exists():
        save_mat_pdf(path)
    return FileResponse(path, media_type="application/pdf", filename="reassemble_capture_mat.pdf")


@app.get("/api/samples")
def samples() -> list[dict]:
    out = []
    if SAMPLES.exists():
        for f in sorted(SAMPLES.iterdir()):
            if (f / "pieces").is_dir():
                t = json.loads((f / "truth.json").read_text()) if (f / "truth.json").exists() else {}
                out.append({"name": f.name, "kind": t.get("kind", "?"),
                            "pieces": len(list((f / "pieces").glob("*.png")))})
    return out


@app.post("/api/jobs/sample")
def create_from_sample(req: SampleRequest) -> dict:
    src = SAMPLES / req.sample
    if not (src / "pieces").is_dir():
        raise HTTPException(404, "unknown sample")
    job = Job(id=uuid.uuid4().hex[:10], folder=RUNS / uuid.uuid4().hex[:10])
    shutil.copytree(src, job.folder)
    start(job, SolveOptions(scorer=req.scorer, ocr_rerank=req.ocr))
    return {"id": job.id}


@app.post("/api/jobs")
async def create_from_upload(files: list[UploadFile] = File(...), mode: str = Form("pieces"),
                             scorer: str = Form("auto")) -> dict:
    """Upload piece images (``mode=pieces``) or one photo/scan to segment (``mode=photo``)."""
    jid = uuid.uuid4().hex[:10]
    folder = RUNS / jid
    (folder / "pieces").mkdir(parents=True)
    if mode == "photo":
        import cv2
        from reassemble.capture.pipeline import capture
        data = await files[0].read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "could not decode image")
        pieces, rep = capture(img)
        for p in pieces:
            save_rgba(folder / "pieces" / f"{p.id}.png", p.image)
    else:
        for i, f in enumerate(files):
            data = await f.read()
            try:
                im = Image.open(io.BytesIO(data)).convert("RGBA")
            except Exception:
                raise HTTPException(400, f"{f.filename} is not an image")
            stem = Path(f.filename or f"piece{i}").stem.replace("/", "_")
            im.save(folder / "pieces" / f"{stem}.png")
    job = Job(id=jid, folder=folder)
    start(job, SolveOptions(scorer=scorer))
    return {"id": jid}


@app.get("/api/jobs/{job_id}")
def job_state(job_id: str) -> dict:
    return get_job(job_id).public()


@app.get("/api/jobs/{job_id}/pieces/{pid}.png")
def piece_png(job_id: str, pid: str) -> Response:
    job = get_job(job_id)
    if pid not in job.pieces:
        raise HTTPException(404)
    buf = io.BytesIO()
    Image.fromarray(job.pieces[pid].image).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


def _doc_rec(job: Job, doc: int) -> Reconstruction:
    return Reconstruction.from_dict(job.documents[doc])


@app.get("/api/jobs/{job_id}/render/{doc}.png")
def render_png(job_id: str, doc: int, heatmap: bool = False) -> Response:
    from reassemble.render.compose import render_grid, render_strips
    job = get_job(job_id)
    rec = _doc_rec(job, doc)
    if job.shape == "grid":
        img = render_grid(job.pieces, rec.grid)
    elif job.shape == "torn":
        from reassemble.solve.torn import TornResult, render_torn
        ids = list(job.pieces)
        plist = [job.pieces[i] for i in ids]
        img = render_torn(plist, TornResult({ids.index(k): tuple(v) for k, v in rec.extra["positions"].items()}, [], [], 0))
    else:
        img = render_strips(job.pieces, rec, heatmap=heatmap)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/jobs/{job_id}/replay.gif")
def replay_gif(job_id: str, doc: int = 0) -> FileResponse:
    from reassemble.render.replay import save_replay_gif
    job = get_job(job_id)
    if job.shape != "strip":
        raise HTTPException(400, "replay is available for strip shreds")
    path = job.folder / f"replay_{doc}_{job.seq}.gif"
    if not path.exists():
        save_replay_gif(job.pieces, _doc_rec(job, doc), path)
    return FileResponse(path, media_type="image/gif", filename="reassemble_replay.gif")


@app.get("/api/jobs/{job_id}/ocr")
def ocr(job_id: str, doc: int = 0, lang: str = "en") -> dict:
    from reassemble.render.compose import render_strips
    from reassemble.verify.ocr import extract_text, tesseract_available
    job = get_job(job_id)
    if not tesseract_available():
        raise HTTPException(503, "tesseract is not installed on the server")
    img = render_strips(job.pieces, _doc_rec(job, doc), background=(255, 255, 255, 255))
    return extract_text(img, lang)


class OrderEdit(BaseModel):
    doc: int = 0
    order: list[str | None]
    client: str | None = None
    # optional: replace the document's constraints too (used by undo/redo)
    locked: list[list[str]] | None = None
    forbidden: list[list[str]] | None = None


@app.post("/api/jobs/{job_id}/order")
async def set_order(job_id: str, edit: OrderEdit) -> dict:
    """A user dragged pieces: accept their order, recompute join costs/confidences."""
    job = get_job(job_id)
    if job.shape != "strip":
        raise HTTPException(400, "editing is available for strip shreds")
    with job.lock:
        sub, D = _doc_cost(job, edit.doc)
        idx = {pid: i for i, pid in enumerate(sub.ids)}
        order = [idx[p] for p in edit.order if p in idx]
        if sorted(order) != list(range(len(sub.ids))):
            raise HTTPException(400, "order must contain every piece of the document exactly once")
        if edit.locked is not None or edit.forbidden is not None:
            job.constraints[edit.doc] = {"locked": edit.locked or [], "forbidden": edit.forbidden or []}
        c = job.constraints.get(edit.doc, {"locked": [], "forbidden": []})
        rec = build_reconstruction(sub, order, D, SolveOptions(detect_gaps=False),
                                   [(idx[a], idx[b]) for a, b in c["locked"] if a in idx and b in idx])
        d = rec.to_dict()
        d.update(doc=edit.doc, pieces=job.documents[edit.doc]["pieces"], method="manual")
        job.documents[edit.doc] = d
        job.seq += 1
        _evaluate(job)
        state = job.public()
    await _broadcast(job, {"type": "state", "state": state, "by": edit.client})
    return state


class ConstraintEdit(BaseModel):
    doc: int = 0
    locked: list[list[str]] = []
    forbidden: list[list[str]] = []
    client: str | None = None


@app.post("/api/jobs/{job_id}/resolve")
async def resolve_with_constraints(job_id: str, edit: ConstraintEdit) -> dict:
    """Lock / split joins and re-solve everything else around them."""
    job = get_job(job_id)
    if job.shape != "strip":
        raise HTTPException(400, "constraints are available for strip shreds")

    def work():
        with job.lock:
            sub = subset(job.prep, [job.prep.ids.index(i) for i in job.documents[edit.doc]["pieces"]])
            opts = SolveOptions(locked=[tuple(x) for x in edit.locked], forbidden=[tuple(x) for x in edit.forbidden])
            rec = solve_prepared(sub, opts)
            d = rec.to_dict()
            d.update(doc=edit.doc, pieces=job.documents[edit.doc]["pieces"])
            job.documents[edit.doc] = d
            job.constraints[edit.doc] = {"locked": edit.locked, "forbidden": edit.forbidden}
            for a, b in edit.locked:
                job.corrections.append({"case": str(job.folder), "left": a, "right": b, "label": 1})
            for a, b in edit.forbidden:
                job.corrections.append({"case": str(job.folder), "left": a, "right": b, "label": 0})
            job.seq += 1
            _evaluate(job)
            return job.public()

    state = await asyncio.to_thread(work)
    await _broadcast(job, {"type": "state", "state": state, "by": edit.client})
    return state


class MoveDoc(BaseModel):
    piece: str
    doc: int
    client: str | None = None


@app.post("/api/jobs/{job_id}/move_doc")
async def move_to_document(job_id: str, edit: MoveDoc) -> dict:
    """Move a piece to another document (mixed bins), then re-solve the documents."""
    job = get_job(job_id)

    def work():
        with job.lock:
            labels = {pid: d["doc"] for d in job.documents for pid in d["pieces"]}
            labels[edit.piece] = edit.doc
            lab = [labels[i] for i in job.prep.ids]
            # relabel to 0..k-1
            uniq = sorted(set(lab))
            lab = [uniq.index(x) for x in lab]
            rec = solve_mixed(job.prep, SolveOptions(), labels=lab)
            job.documents = rec.documents
            job.labels = rec.extra["labels"]
            job.cost_cache.clear()
            job.constraints.clear()
            job.seq += 1
            _evaluate(job)
            return job.public()

    state = await asyncio.to_thread(work)
    await _broadcast(job, {"type": "state", "state": state, "by": edit.client})
    return state


@app.get("/api/jobs/{job_id}/corrections")
def corrections(job_id: str) -> dict:
    """Human-confirmed / rejected joins, ready for ``reassemble finetune``."""
    return {"pairs": get_job(job_id).corrections}


# --------------------------------------------------------------------------- WebSocket


@app.websocket("/ws/jobs/{job_id}")
async def ws_job(ws: WebSocket, job_id: str, name: str = "guest", color: str = "#4ade80") -> None:
    await ws.accept()
    if job_id not in JOBS:
        await ws.send_json({"type": "error", "error": "no such job"})
        await ws.close()
        return
    job = JOBS[job_id]
    cid = uuid.uuid4().hex[:8]
    job.sockets[cid] = (ws, {"id": cid, "name": name[:24], "color": color})
    await ws.send_json({"type": "hello", "client": cid, "state": job.public()})
    await _broadcast(job, {"type": "presence", "online": [i for _, i in job.sockets.values()]})
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "cursor":   # relayed, not stored
                await _broadcast(job, {"type": "cursor", "client": cid, "x": msg.get("x"), "y": msg.get("y"),
                                       "name": name, "color": color}, exclude=cid)
            elif kind == "drag":   # live preview of a piece being dragged by someone else
                await _broadcast(job, {"type": "drag", "client": cid, "piece": msg.get("piece"),
                                       "x": msg.get("x"), "doc": msg.get("doc")}, exclude=cid)
    except WebSocketDisconnect:
        pass
    finally:
        job.sockets.pop(cid, None)
        await _broadcast(job, {"type": "presence", "online": [i for _, i in job.sockets.values()]})
