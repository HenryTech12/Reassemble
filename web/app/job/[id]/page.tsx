"use client";

import { use, useCallback, useEffect, useMemo, useRef, useState } from "react";
import StripBoard, { Cursor, RemoteDrag } from "@/components/StripBoard";
import { API, WS, Doc, Join, JobState, Presence, getJSON, postJSON } from "@/lib/api";

type Snapshot = { doc: number; order: string[]; locked: string[][]; forbidden: string[][] };
type OcrWord = { text: string; conf: number; known: boolean; x: number; y: number; w: number; h: number };

const COLOURS = ["#f472b6", "#a78bfa", "#34d399", "#fbbf24", "#60a5fa", "#fb7185", "#4ade80"];

function useName() {
  const [name, setName] = useState("analyst");
  useEffect(() => {
    try {
      const saved = localStorage.getItem("reassemble:name");
      if (saved) setName(saved);
      else {
        const n = `analyst-${Math.floor(Math.random() * 900 + 100)}`;
        localStorage.setItem("reassemble:name", n);
        setName(n);
      }
    } catch { /* private mode: keep default */ }
  }, []);
  return name;
}

export default function JobPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const name = useName();
  const colour = useMemo(() => COLOURS[Math.floor(Math.random() * COLOURS.length)], []);
  const [state, setState] = useState<JobState | null>(null);
  const [tab, setTab] = useState(0);
  const [heatmap, setHeatmap] = useState(true);
  const [replayKey, setReplayKey] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [online, setOnline] = useState<Presence[]>([]);
  const [cursors, setCursors] = useState<Record<string, Cursor>>({});
  const [drags, setDrags] = useState<Record<string, RemoteDrag>>({});
  const [undo, setUndo] = useState<Snapshot[]>([]);
  const [redo, setRedo] = useState<Snapshot[]>([]);
  const [ocr, setOcr] = useState<{ words: OcrWord[]; readability: Record<string, number> } | null>(null);
  const [wide, setWide] = useState(true);
  const ws = useRef<WebSocket | null>(null);
  const clientId = useRef<string | null>(null);
  const lastCursor = useRef(0);
  const solvedOnce = useRef(false);

  useEffect(() => {
    const f = () => setWide(window.innerWidth >= 768);
    f();
    window.addEventListener("resize", f);
    return () => window.removeEventListener("resize", f);
  }, []);

  const applyState = useCallback((s: JobState) => {
    setState((prev) => {
      if (prev && s.seq < prev.seq) return prev; // server ordering: ignore stale states
      if (s.status === "done" && !solvedOnce.current) {
        solvedOnce.current = true;
        setReplayKey((k) => k + 1); // first result: play the pieces-flying-in animation
      }
      return s;
    });
  }, []);

  // initial fetch + polling fallback until done
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const s = await getJSON<JobState>(`/api/jobs/${id}`);
        applyState(s);
        if (!stop && (s.status === "queued" || s.status === "running")) setTimeout(tick, 800);
      } catch (e) { setErr(String(e)); }
    };
    tick();
    return () => { stop = true; };
  }, [id, applyState]);

  // websocket: live progress, shared edits, presence, cursors
  useEffect(() => {
    if (!name) return;
    const sock = new WebSocket(`${WS}/ws/jobs/${id}?name=${encodeURIComponent(name)}&color=${encodeURIComponent(colour)}`);
    ws.current = sock;
    sock.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.type === "hello") { clientId.current = m.client; applyState(m.state); setOnline(m.state.online); }
      else if (m.type === "state") applyState(m.state);
      else if (m.type === "progress") setState((s) => (s ? { ...s, progress: [...s.progress, m.event].slice(-20) } : s));
      else if (m.type === "presence") setOnline(m.online);
      else if (m.type === "cursor") setCursors((c) => ({ ...c, [m.client]: m }));
      else if (m.type === "drag") {
        setDrags((d) => ({ ...d, [m.client]: m }));
        setTimeout(() => setDrags((d) => { const n = { ...d }; if (n[m.client]?.x === m.x) delete n[m.client]; return n; }), 1500);
      }
    };
    return () => sock.close();
  }, [id, name, colour, applyState]);

  const doc: Doc | undefined = state?.documents[tab];
  const cons = state?.constraints[String(tab)] ?? { locked: [], forbidden: [] };
  const editable = wide && state?.shape === "strip" && state.status === "done";

  function snapshot(): Snapshot | null {
    if (!doc) return null;
    return { doc: tab, order: doc.order.filter(Boolean) as string[], locked: cons.locked, forbidden: cons.forbidden };
  }

  async function run<T>(label: string, f: () => Promise<T>) {
    setBusy(label);
    setErr(null);
    try { return await f(); } catch (e) { setErr(String(e)); } finally { setBusy(null); }
  }

  async function reorder(order: string[], record = true) {
    const snap = snapshot();
    if (record && snap) { setUndo((u) => [...u, snap]); setRedo([]); }
    await run("saving", async () => applyState(await postJSON<JobState>(`/api/jobs/${id}/order`, { doc: tab, order, client: clientId.current })));
  }

  async function joinAction(j: Join, action: "lock" | "split" | "clear") {
    const snap = snapshot();
    if (snap) { setUndo((u) => [...u, snap]); setRedo([]); }
    const same = (p: string[]) => p[0] === j.left && p[1] === j.right;
    let locked = cons.locked.filter((p) => !same(p));
    let forbidden = cons.forbidden.filter((p) => !same(p));
    if (action === "lock") locked = [...locked, [j.left, j.right]];
    if (action === "split") forbidden = [...forbidden, [j.left, j.right]];
    await run("re-solving", async () =>
      applyState(await postJSON<JobState>(`/api/jobs/${id}/resolve`, { doc: tab, locked, forbidden, client: clientId.current })));
  }

  async function restore(from: Snapshot[], setFrom: (s: Snapshot[]) => void, setTo: (f: (s: Snapshot[]) => Snapshot[]) => void) {
    const target = from[from.length - 1];
    const cur = snapshot();
    if (!target || !cur) return;
    setFrom(from.slice(0, -1));
    setTo((s) => [...s, cur]);
    setTab(target.doc);
    await run("restoring", async () => applyState(await postJSON<JobState>(`/api/jobs/${id}/order`, {
      doc: target.doc, order: target.order, locked: target.locked, forbidden: target.forbidden, client: clientId.current })));
  }

  async function moveToDoc(piece: string, to: number) {
    await run("moving", async () => applyState(await postJSON<JobState>(`/api/jobs/${id}/move_doc`, { piece, doc: to, client: clientId.current })));
  }

  async function loadOcr() {
    await run("reading text", async () => setOcr(await getJSON(`/api/jobs/${id}/ocr?doc=${tab}`)));
  }

  function sendCursor(x: number, y: number) {
    const now = Date.now();
    if (now - lastCursor.current < 50 || ws.current?.readyState !== 1) return;
    lastCursor.current = now;
    ws.current.send(JSON.stringify({ type: "cursor", x, y }));
  }
  function sendDrag(piece: string, x: number) {
    if (ws.current?.readyState === 1) ws.current.send(JSON.stringify({ type: "drag", piece, x, doc: tab }));
  }

  // keyboard: ctrl+z / ctrl+shift+z
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (!(e.ctrlKey || e.metaKey) || e.key.toLowerCase() !== "z" || !editable) return;
      e.preventDefault();
      if (e.shiftKey) restore(redo, setRedo, setUndo); else restore(undo, setUndo, setRedo);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  });

  if (!state) return <div className="text-lab-dim font-mono">loading job {id}… {err}</div>;

  const joins = doc?.joins ?? [];
  const avgConf = joins.length ? joins.reduce((a, j) => a + j.confidence, 0) / joins.length : 0;
  const lowConf = joins.filter((j) => !j.blank && j.confidence < 0.25).length;
  const myCursors = Object.values(cursors).filter((c) => c.client !== clientId.current && online.some((o) => o.id === c.client));

  return (
    <div className="space-y-4">
      {/* header */}
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="font-mono text-lg text-white">case <span className="text-lab-accent">{id}</span></h1>
        <span className={`chip ${state.status === "done" ? "text-lab-ok border-lab-ok/50" : state.status === "error" ? "text-lab-bad" : "text-lab-warn"}`}>{state.status}</span>
        <span className="chip">{state.shape}</span>
        {typeof state.extra?.scorer === "string" && <span className="chip">scorer: {state.extra.scorer as string}</span>}
        <div className="flex-1" />
        <div className="flex -space-x-2" title="online now">
          {online.map((o) => (
            <span key={o.id} title={o.name} className="w-7 h-7 rounded-full border-2 border-lab-bg grid place-items-center text-[10px] font-bold text-black" style={{ background: o.color }}>
              {o.name.slice(0, 2).toUpperCase()}
            </span>
          ))}
        </div>
        <button className="btn" onClick={() => navigator.clipboard?.writeText(window.location.href)}>Copy share link</button>
      </div>

      {state.status !== "done" && (
        <div className="panel p-4 font-mono text-sm">
          {state.status === "error" ? <span className="text-lab-bad">{state.error}</span> : (
            <>
              <div className="label mb-2">solver log</div>
              {state.progress.map((p, i) => <div key={i} className="text-lab-dim">[{p.t.toFixed(1)}s] {p.stage}</div>)}
              <div className="animate-pulse text-lab-accent">▮ working…</div>
            </>
          )}
        </div>
      )}

      {state.status === "done" && doc && (
        <div className="grid xl:grid-cols-[1fr_320px] gap-4">
          <div className="space-y-3 min-w-0">
            {state.documents.length > 1 && (
              <div className="flex gap-1 flex-wrap">
                {state.documents.map((d, i) => (
                  <button key={i} onClick={() => { setTab(i); setOcr(null); }} className={`btn ${i === tab ? "border-lab-accent text-white" : ""}`}>
                    Document {i + 1} <span className="text-lab-dim">({d.pieces.length})</span>
                  </button>
                ))}
              </div>
            )}
            <div className="flex flex-wrap gap-2 items-center">
              {state.shape === "strip" && <>
                <label className="btn flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={heatmap} onChange={(e) => setHeatmap(e.target.checked)} /> Confidence heatmap
                </label>
                <button className="btn" onClick={() => setReplayKey((k) => k + 1)}>▶ Replay</button>
                <a className="btn" href={`${API}/api/jobs/${id}/replay.gif?doc=${tab}`} target="_blank" rel="noreferrer">Export GIF</a>
                <button className="btn" disabled={!undo.length || !editable} onClick={() => restore(undo, setUndo, setRedo)}>↶ Undo</button>
                <button className="btn" disabled={!redo.length || !editable} onClick={() => restore(redo, setRedo, setUndo)}>↷ Redo</button>
              </>}
              <a className="btn" href={`${API}/api/jobs/${id}/render/${tab}.png`} target="_blank" rel="noreferrer">Download PNG</a>
              {busy && <span className="text-sm text-lab-accent animate-pulse">{busy}…</span>}
              {err && <span className="text-sm text-lab-bad truncate max-w-md">{err}</span>}
            </div>
            <div className="panel p-3 overflow-hidden">
              {state.shape === "strip" ? (
                <StripBoard jobId={id} doc={doc} sizes={state.pieces} heatmap={heatmap} editable={!!editable}
                  replayKey={replayKey} cursors={myCursors} remoteDrags={Object.values(drags).filter((d) => d.client !== clientId.current)}
                  selected={selected} onSelect={setSelected} onReorder={reorder} onJoin={joinAction}
                  onCursor={sendCursor} onDrag={sendDrag} version={state.seq} />
              ) : (
                // eslint-disable-next-line @next/next/no-img-element
                <img alt="reconstruction" className="max-h-[75vh] mx-auto" src={`${API}/api/jobs/${id}/render/${tab}.png?v=${state.seq}`} />
              )}
            </div>
            {editable && <p className="text-xs text-lab-dim">Drag strips to reorder. Click a seam to lock it (the solver keeps it) or split it (never join these two), then everything else is re-solved around your decision. Ctrl+Z / Ctrl+Shift+Z to undo / redo.</p>}
            {!wide && <p className="text-xs text-lab-warn">Read-only on small screens — open on a laptop to edit.</p>}
          </div>

          <aside className="space-y-3">
            <div className="panel p-4 space-y-2">
              <div className="label">assessment</div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div><div className="text-2xl font-mono text-white">{(avgConf * 100).toFixed(0)}%</div><div className="text-xs text-lab-dim">mean join confidence</div></div>
                <div><div className="text-2xl font-mono text-white">{lowConf}</div><div className="text-xs text-lab-dim">joins to review</div></div>
                <div><div className="text-2xl font-mono text-white">{doc.gaps?.length ?? 0}</div><div className="text-xs text-lab-dim">missing pieces?</div></div>
                <div><div className="text-2xl font-mono text-white">{doc.pieces.length}</div><div className="text-xs text-lab-dim">pieces</div></div>
              </div>
              {state.metrics && (
                <div className="pt-2 border-t border-lab-line text-xs font-mono text-lab-dim space-y-0.5">
                  <div className="label mb-1">vs ground truth</div>
                  {Object.entries(state.metrics).filter(([, v]) => typeof v === "number").map(([k, v]) => (
                    <div key={k} className="flex justify-between"><span>{k}</span><span className="text-white">{Number(v).toFixed(3)}</span></div>
                  ))}
                </div>
              )}
              <div className="flex gap-3 text-[11px] text-lab-dim pt-1">
                <span><i className="inline-block w-2 h-2 bg-lab-ok mr-1" />sure</span>
                <span><i className="inline-block w-2 h-2 bg-lab-warn mr-1" />unsure</span>
                <span><i className="inline-block w-2 h-2 bg-lab-bad mr-1" />doubtful/gap</span>
                <span><i className="inline-block w-2 h-2 bg-sky-400 mr-1" />locked</span>
              </div>
            </div>

            {selected && (
              <div className="panel p-4 space-y-2 text-sm">
                <div className="label">selected piece</div>
                <div className="font-mono text-white">{selected}</div>
                {state.documents.length > 1 && editable && (
                  <div className="flex items-center gap-2">
                    <span className="text-lab-dim text-xs">move to</span>
                    <select className="bg-lab-bg border border-lab-line rounded px-2 py-1 text-sm" value={tab}
                      onChange={(e) => moveToDoc(selected, state.documents[Number(e.target.value)].doc)}>
                      {state.documents.map((d, i) => <option key={i} value={i}>Document {i + 1}</option>)}
                    </select>
                  </div>
                )}
              </div>
            )}

            {state.shape === "strip" && (
              <div className="panel p-4 space-y-2">
                <div className="flex items-center"><div className="label flex-1">extracted text (OCR)</div>
                  <button className="btn" onClick={loadOcr} disabled={!!busy}>{ocr ? "Refresh" : "Read text"}</button></div>
                {ocr && (
                  <>
                    <div className="text-xs text-lab-dim">readability {(ocr.readability.score * 100).toFixed(0)}% · {(ocr.readability.dict_fraction * 100).toFixed(0)}% dictionary words</div>
                    <div className="text-sm leading-relaxed max-h-72 overflow-y-auto">
                      {ocr.words.map((w, i) => (
                        <span key={i} className={w.known ? "" : "bg-lab-bad/25 text-red-200 rounded px-0.5"} title={w.known ? "" : `uncertain (conf ${w.conf.toFixed(0)})`}>{w.text} </span>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}

            <div className="panel p-4 text-xs text-lab-dim space-y-1">
              <div className="label">collaborators</div>
              {online.map((o) => <div key={o.id}><span className="inline-block w-2 h-2 rounded-full mr-2" style={{ background: o.color }} />{o.name}{o.id === clientId.current ? " (you)" : ""}</div>)}
              <a className="underline hover:text-white" href={`${API}/api/jobs/${id}/corrections`} target="_blank" rel="noreferrer">export corrections for fine-tuning</a>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
