"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { API, Doc, Join, confColour } from "@/lib/api";

export type Cursor = { client: string; name: string; color: string; x: number; y: number };
export type RemoteDrag = { client: string; piece: string; x: number };

type Props = {
  jobId: string;
  doc: Doc;
  sizes: Record<string, { w: number; h: number }>;
  heatmap: boolean;
  editable: boolean;
  replayKey: number;
  cursors: Cursor[];
  remoteDrags: RemoteDrag[];
  selected: string | null;
  onSelect: (pid: string | null) => void;
  onReorder: (order: string[]) => void;
  onJoin: (join: Join, action: "lock" | "split" | "clear") => void;
  onCursor: (x: number, y: number) => void;
  onDrag: (piece: string, x: number) => void;
  version: number;
};

const DISPLAY_H = 560;

/** The reconstruction canvas: pieces laid out left-to-right, seams coloured by confidence. */
export default function StripBoard(p: Props) {
  const boardRef = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<{ id: string; dx: number; startX: number } | null>(null);
  const [menu, setMenu] = useState<{ join: Join; x: number } | null>(null);
  const [flying, setFlying] = useState(false);

  const maxH = Math.max(...Object.values(p.sizes).map((s) => s.h), 1);
  const s = DISPLAY_H / maxH;
  const medW = useMemo(() => {
    const ws = Object.values(p.sizes).map((x) => x.w).sort((a, b) => a - b);
    return ws[Math.floor(ws.length / 2)] ?? 20;
  }, [p.sizes]);

  // layout: cumulative x in display pixels
  const layout = useMemo(() => {
    let x = 0;
    return p.doc.order.map((id, i) => {
      const w = (id ? p.sizes[id]?.w ?? medW : medW) * s;
      const item = { id, x, w, i };
      x += w;
      return item;
    });
  }, [p.doc.order, p.sizes, s, medW]);
  const totalW = layout.length ? layout[layout.length - 1].x + layout[layout.length - 1].w : 0;
  const joinByRight = useMemo(() => Object.fromEntries(p.doc.joins.map((j) => [j.right, j])), [p.doc.joins]);

  // replay: pieces start scattered, then fly into place one after another in solve order
  useEffect(() => {
    if (!p.replayKey) return;
    setFlying(true);
    const t = setTimeout(() => setFlying(false), 60);
    return () => clearTimeout(t);
  }, [p.replayKey]);
  const scatter = useMemo(() => {
    const r: Record<string, { x: number; y: number; a: number }> = {};
    let seed = p.replayKey * 9973 + 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    p.doc.order.forEach((id) => { if (id) r[id] = { x: rnd() * totalW, y: 120 + rnd() * 200, a: (rnd() - 0.5) * 70 }; });
    return r;
  }, [p.replayKey, p.doc.order, totalW]);

  function pointerDown(e: React.PointerEvent, id: string) {
    p.onSelect(id);
    if (!p.editable) return;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    setDrag({ id, dx: 0, startX: e.clientX });
    setMenu(null);
  }
  function pointerMove(e: React.PointerEvent) {
    const r = boardRef.current?.getBoundingClientRect();
    if (r) p.onCursor((e.clientX - r.left) / Math.max(r.width, 1), (e.clientY - r.top) / Math.max(r.height, 1));
    if (!drag) return;
    const dx = e.clientX - drag.startX;
    setDrag({ ...drag, dx });
    const item = layout.find((l) => l.id === drag.id);
    if (item && totalW) p.onDrag(drag.id, (item.x + dx) / totalW);
  }
  function pointerUp() {
    if (!drag) return;
    const item = layout.find((l) => l.id === drag.id);
    if (item && Math.abs(drag.dx) > 3) {
      const centre = item.x + drag.dx + item.w / 2;
      const others = layout.filter((l) => l.id && l.id !== drag.id);
      let idx = others.findIndex((l) => l.x + l.w / 2 > centre);
      if (idx < 0) idx = others.length;
      const order = others.map((l) => l.id as string);
      order.splice(idx, 0, drag.id);
      p.onReorder(order);
    }
    setDrag(null);
  }

  return (
    <div className="overflow-x-auto pb-3">
      <div
        ref={boardRef}
        className="relative select-none mx-auto"
        style={{ width: totalW + 2, height: DISPLAY_H }}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onClick={(e) => { if (e.target === boardRef.current) { p.onSelect(null); setMenu(null); } }}
      >
        {layout.map((l, k) => {
          if (!l.id) {
            return (
              <div key={`gap-${k}`} title="Probable missing piece"
                className="absolute top-0 border-2 border-lab-bad/70"
                style={{ left: l.x, width: l.w, height: DISPLAY_H,
                  background: "repeating-linear-gradient(45deg, rgba(239,68,68,.35) 0 6px, rgba(60,10,10,.6) 6px 12px)" }} />
            );
          }
          const id = l.id;
          const isDrag = drag?.id === id;
          const remote = p.remoteDrags.find((r) => r.piece === id);
          const sc = scatter[id];
          const inFlight = flying && sc;
          const tx = inFlight ? sc.x - l.x : isDrag ? drag!.dx : remote ? remote.x * totalW - l.x : 0;
          const ty = inFlight ? sc.y : isDrag ? -10 : 0;
          return (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={id}
              src={`${API}/api/jobs/${p.jobId}/pieces/${id}.png`}
              alt={id}
              draggable={false}
              onPointerDown={(e) => pointerDown(e, id)}
              className={`absolute top-0 ${p.editable ? "cursor-grab active:cursor-grabbing" : ""} ${p.selected === id ? "outline outline-2 outline-lab-accent z-10" : ""} ${remote ? "outline outline-2 outline-dashed outline-fuchsia-400" : ""}`}
              style={{
                left: l.x, width: l.w, height: (p.sizes[id]?.h ?? maxH) * s,
                transform: `translate(${tx}px, ${ty}px) rotate(${inFlight ? sc.a : 0}deg)`,
                transition: isDrag || inFlight ? "none" : `transform ${p.replayKey ? 0.7 : 0.25}s cubic-bezier(.2,.8,.2,1) ${p.replayKey && !flying ? k * 0.06 : 0}s`,
                zIndex: isDrag ? 20 : undefined,
                filter: isDrag ? "drop-shadow(0 8px 12px rgba(0,0,0,.6))" : undefined,
              }}
            />
          );
        })}
        {/* seams */}
        {layout.map((l) => {
          const j = l.id ? joinByRight[l.id] : undefined;
          if (!j) return null;
          const col = confColour(j);
          return (
            <button
              key={`seam-${j.left}-${j.right}`}
              title={`${j.left} → ${j.right}  confidence ${(j.confidence * 100).toFixed(0)}%${j.locked ? " (locked)" : ""}${j.ocr_suspect ? `  ·  ${j.ocr_suspect} odd word(s) across` : ""}`}
              onClick={(e) => { e.stopPropagation(); setMenu({ join: j, x: l.x }); }}
              className="absolute top-0 z-[5] group"
              style={{ left: l.x - 4, width: 8, height: DISPLAY_H }}
            >
              <span className="block mx-auto h-full transition-opacity"
                style={{ width: j.locked ? 4 : 6, background: col, opacity: p.heatmap || j.locked ? 0.55 : 0 }} />
              <span className="absolute inset-0 bg-white/0 group-hover:bg-white/20" />
            </button>
          );
        })}
        {menu && (
          <div className="absolute z-30 panel p-2 text-sm shadow-xl" style={{ left: Math.min(menu.x + 8, totalW - 160), top: 16 }}>
            <div className="font-mono text-xs text-lab-dim mb-1">{menu.join.left} | {menu.join.right}</div>
            <div className="text-xs mb-2">confidence <b style={{ color: confColour(menu.join) }}>{(menu.join.confidence * 100).toFixed(0)}%</b></div>
            {p.editable ? (
              <div className="flex flex-col gap-1">
                <button className="btn" onClick={() => { p.onJoin(menu.join, "lock"); setMenu(null); }}>🔒 Lock join</button>
                <button className="btn" onClick={() => { p.onJoin(menu.join, "split"); setMenu(null); }}>✂ Split here</button>
                <button className="btn" onClick={() => { p.onJoin(menu.join, "clear"); setMenu(null); }}>Clear constraint</button>
              </div>
            ) : <div className="text-xs text-lab-dim">read-only view</div>}
            <button className="text-xs text-lab-dim mt-1 hover:text-white" onClick={() => setMenu(null)}>close</button>
          </div>
        )}
        {p.cursors.map((c) => (
          <div key={c.client} className="absolute z-40 pointer-events-none" style={{ left: c.x * (totalW + 2), top: c.y * DISPLAY_H }}>
            <svg width="16" height="16" viewBox="0 0 16 16"><path d="M0 0 L12 5 L6 7 L4 13 Z" fill={c.color} /></svg>
            <span className="text-[10px] px-1 rounded text-black" style={{ background: c.color }}>{c.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
