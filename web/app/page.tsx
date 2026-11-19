"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { API, getJSON, postJSON } from "@/lib/api";

type Sample = { name: string; kind: string; pieces: number };

export default function Home() {
  const router = useRouter();
  const [samples, setSamples] = useState<Sample[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [mode, setMode] = useState<"pieces" | "photo">("pieces");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [over, setOver] = useState(false);

  useEffect(() => {
    getJSON<Sample[]>("/api/samples").then(setSamples).catch(() => setErr(`API not reachable at ${API} — run "make api".`));
  }, []);

  async function upload() {
    setBusy(true);
    setErr(null);
    try {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      fd.append("mode", mode);
      const r = await fetch(API + "/api/jobs", { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      router.push(`/job/${(await r.json()).id}`);
    } catch (e) {
      setErr(String(e));
      setBusy(false);
    }
  }

  async function trySample(name: string) {
    setBusy(true);
    try {
      const { id } = await postJSON<{ id: string }>("/api/jobs/sample", { sample: name });
      router.push(`/job/${id}`);
    } catch (e) {
      setErr(String(e));
      setBusy(false);
    }
  }

  return (
    <div className="grid lg:grid-cols-[1.3fr_1fr] gap-6">
      <section className="panel p-6">
        <p className="label mb-2">evidence intake</p>
        <h1 className="text-2xl font-semibold text-white mb-1">Reassemble a shredded document</h1>
        <p className="text-lab-dim text-sm mb-5">
          Drop scans of individual strips, or one photo of strips laid on the{" "}
          <a className="underline hover:text-white" href={`${API}/api/mat.pdf`}>capture mat</a>. The solver scores every pair of
          edges, orders the pieces, and hands you an editor to fix what it got wrong.
        </p>
        <div className="flex gap-2 mb-3">
          {(["pieces", "photo"] as const).map((m) => (
            <button key={m} onClick={() => setMode(m)} className={`btn ${mode === m ? "border-lab-accent text-white" : ""}`}>
              {m === "pieces" ? "Piece images" : "One photo / scan"}
            </button>
          ))}
        </div>
        <label
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); setFiles(Array.from(e.dataTransfer.files)); }}
          className={`block border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition ${over ? "border-lab-accent bg-sky-500/5" : "border-lab-line hover:border-lab-dim"}`}
        >
          <input type="file" multiple={mode === "pieces"} accept="image/*" className="hidden"
            onChange={(e) => setFiles(Array.from(e.target.files ?? []))} />
          <div className="font-mono text-lab-accent text-3xl mb-2">⇣</div>
          <div className="text-sm">{files.length ? `${files.length} file(s) ready` : "Drag & drop images here, or click to choose"}</div>
          <div className="text-xs text-lab-dim mt-1">PNG with transparency works best for pieces; JPG is fine for photos.</div>
        </label>
        <div className="mt-4 flex items-center gap-3">
          <button className="btn btn-primary" disabled={!files.length || busy} onClick={upload}>
            {busy ? "Starting…" : "Reconstruct"}
          </button>
          {err && <span className="text-sm text-lab-bad">{err}</span>}
        </div>
      </section>

      <section className="panel p-6">
        <p className="label mb-2">or try a sample</p>
        <p className="text-sm text-lab-dim mb-4">Synthetic cases with known ground truth, so you also get accuracy numbers.</p>
        {samples.length === 0 && <p className="text-sm text-lab-dim">No samples yet — run <code className="font-mono">make demo-synth</code>.</p>}
        <ul className="divide-y divide-lab-line">
          {samples.map((s) => (
            <li key={s.name} className="flex items-center gap-3 py-2">
              <span className="font-mono text-sm text-white flex-1">{s.name}</span>
              <span className="chip">{s.kind}</span>
              <span className="text-xs text-lab-dim w-16 text-right">{s.pieces} pcs</span>
              <button className="btn" disabled={busy} onClick={() => trySample(s.name)}>Try</button>
            </li>
          ))}
        </ul>
      </section>

      <p className="lg:col-span-2 text-xs text-lab-dim">
        For recovering your own documents, testing your organisation&apos;s shredders, archives and education only. Don&apos;t
        use it on documents you have no right to access.
      </p>
    </div>
  );
}
