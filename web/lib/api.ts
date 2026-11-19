export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const WS = API.replace(/^http/, "ws");

export type Join = {
  left: string; right: string; cost: number; confidence: number;
  gap?: boolean; locked?: boolean; blank?: boolean; ocr_suspect?: number; side?: string;
};
export type Doc = {
  doc: number; order: (string | null)[]; pieces: string[]; joins: Join[];
  gaps: { after: string; before: string; est_width: number }[];
  grid?: (string | null)[][] | null; method?: string; score?: number;
  extra?: Record<string, unknown>;
};
export type Presence = { id: string; name: string; color: string };
export type JobState = {
  id: string; status: "queued" | "running" | "done" | "error"; shape: "strip" | "grid" | "torn";
  error: string | null; seq: number; progress: { stage: string; t: number }[];
  documents: Doc[]; labels: Record<string, number>;
  constraints: Record<string, { locked: string[][]; forbidden: string[][] }>;
  pieces: Record<string, { w: number; h: number }>; online: Presence[];
  metrics?: Record<string, number> | null; extra?: Record<string, unknown>;
};

export async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(API + path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(API + path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export function confColour(j?: Join): string {
  if (!j) return "#64748b";
  if (j.gap) return "#ef4444";
  if (j.locked) return "#38bdf8";
  if (j.blank) return "#64748b";
  if (j.confidence >= 0.6) return "#22c55e";
  if (j.confidence >= 0.25) return "#eab308";
  return "#ef4444";
}
