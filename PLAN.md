# Reassemble — build plan

Reassemble takes scans/photos of shredded paper and digitally rebuilds the original
document. It is built in phases; each phase is finished, tested, demoed and committed
before the next begins.

## Approach in one paragraph
Everything is driven by a **synthetic data generator** (Phase 0) that renders realistic
pages and shreds them with a known ground truth, so every later algorithm can be measured.
Pieces are RGBA images (alpha = paper mask) so irregular/torn edges flow through the same
code. Matching is a classic *compatibility matrix* problem: score every ordered pair of
edges (handcrafted prediction-based cost, later a small CNN), then solve an ordering /
placement problem (greedy best-buddies, beam search, asymmetric TSP with 2-opt/or-opt;
greedy grid placement for cross-cut). OCR (Tesseract) is used as a global sanity check to
rerank candidates. A FastAPI backend + Next.js editor adds the human in the loop.

## Phases
| # | Phase | Key deliverable | Done when |
|---|-------|-----------------|-----------|
| 0 | Synthetic data | page renderer + strip / cross-cut / torn / double-sided / mixed shredders | `make demo-synth` writes a sample of every case |
| 1 | Strip MVP | features, pairwise cost, greedy + beam + ATSP solvers, gap detection, metrics, CLI | ≥95 % neighbour accuracy on clean 30-strip pages |
| 2 | Quality | OCR rerank, two-sided cost, mixed-document clustering, gap placeholders | ≥90 % purity on 3-page bins; OCR rerank before/after |
| 3 | Learned matcher | small CPU-trainable CNN on edge pairs, scorer comparison table | README table: handcrafted vs learned vs combined |
| 4 | Harder shreds | cross-cut 2D solver, torn-edge shape matching, handwriting/photos | accuracy reported per piece size |
| 5 | Real-world capture | photo → clean RGBA pieces; simulated test photo; capture mat PDF | `capture` then `solve` works end-to-end |
| 6 | API + web | FastAPI + WebSocket, Next.js editor with heatmap, replay, constrained re-solve, collab | runs locally |
| 7 | Product angles | shredder security audit PDF, archive batch mode | report generated |
| 8 | Polish | README, diagrams, Docker, `make demo` | — |

## Conventions
* Python 3.11+ package `reassemble/`, tests in `tests/` (pytest), `Makefile` for all commands.
* Case folder format: `pieces/<id>.png` (RGBA), optional `back/<id>.png`, `truth.json`.
* Piece ids are random-looking (`p017`), so file names never leak the true order.
* Every algorithm gets a docstring explaining *why* it works.
* If something is much harder than expected: ship a simpler working version, document the
  limitation in the README, move on.
