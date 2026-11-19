# Reassemble

Reassemble takes scans or photos of shredded paper and rebuilds the original document
digitally. Under the hood it uses computer vision to score how well piece edges fit, a
combinatorial solver to order the pieces, and OCR to check the result. A web editor lets a
person correct what the solver gets wrong. The idea comes from the 2011 DARPA Shredder
Challenge.

> **Ethics.** Reassemble is meant for recovering your own documents, testing your
> organisation's shredders, archive work and teaching. Don't use it on documents you have
> no right to access.

## Quick start

```bash
make setup          # venv + Python deps (needs `uv`; also: sudo apt-get install tesseract-ocr)
make test           # run the test suite
make demo-synth     # generate every synthetic test case into samples/
reassemble solve samples/memo_30_noisy --out result/   # or: python -m reassemble.cli solve ...
reassemble bench    # accuracy benchmark (--suite strips|mixed|double|ocr|crosscut|torn|scorers)
reassemble train    # retrain the CNN edge matcher (~2.5 min on CPU)
reassemble solve samples/mixed_3 --out result/mixed --ocr   # mixed bin + OCR rerank
```

`solve` writes `reconstruction.png`, `heatmap.png` (joins coloured by confidence) and
`result.json`.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Synthetic data generator + digital shredders | ✅ |
| 1 | Strip reconstruction MVP | ✅ |
| 2 | OCR verification & rerank, two-sided matching, mixed-document separation, gap placeholders | ✅ |
| 3 | Learned CNN edge matcher (+ fine-tuning from human corrections) | ✅ |
| 4 | Cross-cut 2D solver, torn-paper shape matching, handwriting/photos | ✅ (cross-cut: partial) |
| 5–8 | capture, web app, product angles, polish | in progress |

## How it works (strip shreds)

1. **Normalise each piece.** Estimate the strip's long axis with PCA on the alpha mask,
   rotate it upright, and crop it.
2. **Read the edges.** For each row, find where the paper actually ends (from the alpha
   mask) and read the pixels just inside that point, so frayed or ragged edges still work.
   Colours are white-balanced against the paper tone. If the document looks blurry, edges
   are also sharpened with an unsharp mask (see the results below for why).
3. **Score every ordered pair (A, B).** The main cue is a *prediction-based* seam cost
   (Pomeranz et al.): extrapolate A's last two columns linearly and compare the result with
   B's first column, and do the same from B towards A. The per-row error ignores a band the
   size of the estimated noise, so blank rows don't swamp the signal. Two weak cues are
   added: text-line alignment, weighted by how regular the text lines are, and paper/ink
   colour.
4. **Solve.** The task is to find a Hamiltonian path, which becomes an **asymmetric TSP**
   once a virtual "page border" node is added. A piece is cheap to start or end on when its
   outer edge is blank. Three solvers run:
   * *greedy best-buddies*: link mutual best matches first, then cheapest-first without
     creating cycles (union-find);
   * *beam search* with Held–Karp-style state merging;
   * *ATSP local search* (vectorised Or-opt + asymmetric 2-opt), seeded from both of the
     above.

   The cheapest tour wins. Blank margin strips carry no information, so they are taken out
   of the optimisation and placed at the page edges in whatever way balances the margins.
5. **Confidence per join** is a *min-marginal*: how much the best solution's cost rises if
   that join is forbidden and the rest is re-optimised. A join can look only slightly better
   than the alternatives locally and still be forced by the rest of the solution, so this
   works far better than a local best-vs-second-best ratio.
6. **Gap detection** flags a join as a probable missing strip when its cost is ≥1.5× the
   median join cost *and* neither piece prefers the other (ratio-normalised cost > 1.2).

## Results

Benchmark: `reassemble bench`. Each run renders fresh pages (5 per page type, seeds
100–104), shreds them into 30 strips and solves them blind.

* **neighbour**: fraction of true adjacent pairs that are reproduced.
* **direct**: fraction of strips at exactly the right index.
* ***_content***: the same metrics with **blank strips ignored**. The left and right
  margins of a page produce 3–4 completely blank strips whose order cannot be determined
  and does not affect the picture (standard practice in the strip-shredding literature).

**Clean scans, 30 strips** (target ≥ 95 % ✅)

| page | neighbour | neighbour (content) | direct (content) |
|---|---|---|---|
| memo | 0.883 | 0.976 | 0.915 |
| letter | 0.910 | 0.976 | 0.800 |
| invoice | 0.883 | 0.961 | 0.785 |
| photo | 0.931 | 1.000 | 1.000 |
| handwriting | 0.848 | 0.984 | 0.600 |
| **all** | **0.891** | **0.979** | 0.820 |

**Realistic scans** (±0.4° rotation, σ=6 noise, blur σ=0.7, ±6 % brightness, ±8 % width,
3 px ragged cut)

| page | neighbour | neighbour (content) | direct (content) |
|---|---|---|---|
| memo | 0.903 | 0.968 | 0.946 |
| letter | 0.869 | 1.000 | 1.000 |
| invoice | 0.779 | 0.843 | 0.420 |
| photo | 0.890 | 1.000 | 1.000 |
| handwriting | 0.828 | 0.992 | 0.800 |
| **all** | **0.854** | **0.961** | 0.833 |

Solvers compared (neighbour accuracy, all pages):

| solver | clean | realistic |
|---|---|---|
| greedy best-buddies | 0.829 | 0.757 |
| beam search (w=200) | 0.801 | 0.739 |
| ATSP Or-opt/2-opt (kept) | **0.892** | **0.861** |

Runtime is about 0.6–0.8 s per 30-strip page on a laptop CPU, including feature extraction
and confidence computation.

**What made the difference** while tuning (realistic set, content-neighbour accuracy):

| change | before → after |
|---|---|
| model fraying as a *shared* ragged cut line instead of material lost on both sides | 0.05 → 0.95 top-1 match rate on frayed strips |
| white-balance edges by paper colour | 0.37 → 0.93 top-1 with brightness jitter |
| unsharp-mask before matching (only when the scan is blurry) | 0.63 → 0.96 |
| solve on raw costs instead of ratio-normalised ones | ratios of near-zero costs are unstable on blank edges |

## Phase 2: quality features

**OCR verification** (`--ocr`, Tesseract). The pixel matcher only ever sees one seam at a
time, while OCR sees the whole page. The candidate list is the best order plus, for each
join, the re-optimised order with that join forbidden (the same computation as the
confidence), deduplicated and cut to 8. Each candidate is rendered and OCR'd in parallel and
scored by *readability* = 0.6 × fraction of tokens that are dictionary words + 0.4 × mean
OCR confidence. The winner maximises `readability − α·(extra seam cost)`. Joins crossed by
several non-words are flagged as `ocr_suspect` for the editor. Word lists live in
`reassemble/verify/wordlists/`, and you choose them with `--lang en,pcm,yo` (Nigerian
Pidgin and Yoruba ship as small starter lists).

| noisy pages (memo, letter, invoice, handwriting × 6 seeds) | neighbour | neighbour (content) | direct (content) |
|---|---|---|---|
| without OCR | 0.813 | 0.916 | 0.757 |
| **with OCR rerank** | **0.833** | **0.939** | **0.796** |

OCR adds about 2–3 s per page. The best candidate in the list would score 0.941, so OCR
picks almost optimally; what limits it is how good the candidate list is.

**Two-sided matching.** When back images exist, the back-side seam cost is added. The back
is stored as you would read it, so "B is right of A on the front" means "A.back is right of
B.back": the back cost matrix is simply **transposed**. Noisy 40-strip double-sided pages,
10 seeds:

| | neighbour | neighbour (content) | direct (content) |
|---|---|---|---|
| front only | 0.759 | 0.864 | 0.485 |
| **front + back** | **0.849** | **0.976** | **0.618** |

**Mixed-document separation** (`solve --mixed`, automatic for mixed cases). All strips of
one page have their text lines at the same heights, so the main cue is the correlation of
smoothed "inside a text line" row profiles: about 0.25 distance within a page versus about
0.9 across pages. Strips are linked to their 4 nearest neighbours when the distance is
< 0.55. Connected components become documents, so the number of documents doesn't have to
be guessed; components that are still similar on median get merged. Blank strips are
assigned by *paper chromaticity* (r,g,b)/(r+g+b), which ignores scan brightness but keeps
the paper's tint. Each document is then solved on its own. Noisy bins, 8 seeds each:

| bin | purity | #docs found correctly | per-document neighbour (content) |
|---|---|---|---|
| 2 pages | 0.981 | 75 % | 0.817 |
| **3 pages** | **0.949** ✅ | 75 % | 0.800 |
| 5 pages | 0.890 | 63 % | 0.805 |

(Silhouette-based agglomerative clustering, the first attempt, got purity of only about
1/k: one giant cluster plus singletons.)

**Missing pieces** show up in renders as red hatched placeholders of median strip width.

## Phase 3: learned edge matcher

`reassemble/match/learned.py` defines a small CNN (4 conv layers, about 70k parameters).
It looks at a **seam patch**: 32 rows × (last 8 px of A + first 8 px of B) × RGB, read
relative to the paper boundary. It outputs P(true neighbours). This is the early-fusion
design from Paixão et al.: the convolutions see both sides at once, so they learn stroke
continuity directly. A pair's cost is the mean −log P over its *informative* patches (those
with some ink); blank patches are skipped.

* **Data comes free from the generator.** 80 random pages, each shredded into 20–45 strips
  with random realism (60 % noisy, 30 % extra blur). Positives are true neighbours.
  Negatives are *hard*: other strips of the same page, including near-misses 2–3 strips
  away. Training augments with ±1 px vertical shifts and brightness jitter.
* **Held-out fonts.** Training uses Liberation Serif/Sans/Mono, DejaVu Sans, FreeSerif and
  Caveat. Validation uses Caladea, Carlito, DejaVu Serif and Indie Flower.
* **CPU-trainable.** `reassemble train` takes about **2.5 minutes** on 4 CPU cores
  (77k training patches, 6 epochs, about 81 % single-patch validation accuracy). Weights are
  in `models/edge_cnn.pt` (280 KB).
* `--scorer auto` (the default) uses the CNN when PyTorch and the weights are available,
  and the handcrafted scorer otherwise.
* **Human-in-the-loop hook:** `reassemble finetune corrections.json` fine-tunes on pairs
  exported from the editor: label 1 for locked joins, 0 for split ones. Human pairs are
  upweighted ×4 and mixed with replayed synthetic data so the model doesn't forget.

**Scorer comparison** (content-neighbour accuracy, 4 pages per condition, seeds 700–703):

| condition | handcrafted | learned | combined |
|---|---|---|---|
| clean | 1.000 | 1.000 | 0.992 |
| realistic noise | 0.858 | **1.000** | 1.000 |
| heavy noise + blur (σ=10, blur 1.2) | 0.404 | **1.000** | 0.953 |
| handwriting (noisy) | 1.000 | 1.000 | 1.000 |
| 60 strips (noisy) | 0.684 | **1.000** | 0.990 |
| **held-out fonts** (noisy) | 0.910 | **1.000** | 1.000 |

With the learned scorer, the standard benchmark (`reassemble bench --scorer learned`,
4 pages × 5 types) reaches **1.000 content-neighbour accuracy** on clean, noisy and
noisy-with-2-missing-strips pages. Gap detection improves from 0.14 recall / 0.20 precision
to **0.40 / 0.70**. The cost is speed: about 3 s per 30-strip page on CPU instead of
about 0.7 s. "Combined" (average of the two normalised costs) never beat the CNN alone.

## Phase 4: harder shred types

**Cross-cut (2D)**, `reassemble/solve/grid.py`. Top/bottom costs come from the *same*
matcher applied to **transposed** pieces. Two placers run, and the cheaper result wins:
* a Pomeranz/Paikin-style greedy placer on a floating grid, which fills the most confident
  slot first and prefers slots with more placed neighbours;
* *rows-then-stack*: an ATSP with `rows` copies of the border node splits all pieces into
  rows, then whole rows are stacked using the **summed** top/bottom cost down every column.

A swap/cyclic-shift local search refines both. Costs are ratio-normalised, which matters
in 2D: with raw costs, wrong layouts that push every inky edge onto the outer border scored
better than the truth. Clean letters and memos, 3 seeds each:

| grid | piece size | neighbour | neighbour (content) | direct | time |
|---|---|---|---|---|---|
| 3×2 | 425×366 px | 0.952 | 0.952 | 0.833 | 0.25 s |
| 4×3 | 283×275 px | 0.804 | 0.810 | 0.167 | 0.28 s |
| 6×5 | 170×183 px | 0.595 | 0.635 | 0.356 | 0.36 s |
| 8×6 | 141×137 px | 0.402 | 0.429 | 0.056 | 0.52 s |
| 10×8 | 106×110 px | 0.270 | 0.318 | 0.015 | 0.91 s |

Honest assessment: this is the weakest part of the project. Left/right matching is decent
(top-1 about 0.75), but **top/bottom matching is poor** (top-1 about 0.2). About half of
all horizontal cuts fall in the blank space between text lines, so there's nothing to
match. A text-line-*phase* cue is implemented (`line_phase_cost`) but gave no consistent
gain, so it is off by default. The CNN wasn't better on cross-cut either; it was trained
only on vertical seams.

**Torn paper**, `reassemble/solve/torn.py`. A tear is a shared curve. If B sits at offset
(dx, dy) to the right of A, then `x_A(y) + 1 − x_B(y − dy)` must be constant across the
overlapping rows. We search every dy, take dx as the median, and use the trimmed mean
absolute deviation as the **shape cost**. Colours read 4 px inside each boundary (past the
white fibre rim) add a content-continuity term. Rows lying on the page's own straight edge
are excluded. Pieces are assembled Kruskal-style by ratio-normalised match quality, and any
merge that would make pieces overlap is rejected.

| torn pages (memo, letter, photo, handwriting × 3 seeds) | 2×4 pieces | 3×4 pieces |
|---|---|---|
| clean, 14 px tears | **1.000** | **1.000** |
| clean, subtle 6 px tears | 1.000 | 1.000 |
| noisy (no rotation) | 1.000 | 1.000 |
| noisy + ±0.4° rotation | 0.558 | 0.613 |

**Handwriting and photos.** The baseline cue is weighted by each strip's text-line
regularity, so it fades out for handwriting and photos, and the pixel/CNN seam cost carries
those cases (handwriting: 1.000 with either scorer; photos: 1.000 content accuracy).

## Known limitations (honest list)

* **Blank strips** can't be ordered. They are placed at the margins, but which blank strip
  goes where is a coin toss, which is why raw neighbour accuracy sits around 0.85–0.9.
* **Monospace fonts** are hard. In a monospaced page, the whitespace between characters
  lines up vertically across every line, so many cuts land in blank columns (about 0.68
  content accuracy on that seed).
* **Gap (missing strip) detection is weak**: about 0.15–0.6 recall depending on the page
  type. It works on handwriting, but on smooth photos the gradient extrapolates across a
  missing strip almost seamlessly.
* Noisy **invoices** are the hardest printed case (0.84). Their table rules repeat the same
  pattern on many strips.
* **Torn pieces must be upright.** The shape matcher has no rotation search, so ±0.4° of
  rotation drops accuracy to about 0.6. (Strips are deskewed by PCA; torn pieces have no
  long axis to deskew by.)
* **Cross-cut beyond about 6×5 pieces** is mostly wrong (see the table above).
* Mixed bins: two pages built from the *same template* (two memos) have nearly identical
  line layouts and can merge into one cluster. The number of documents is right about 75 %
  of the time; when it's wrong, it's usually over-segmentation.

## Project layout

```
reassemble/
  synth/      page renderer (memo, letter, invoice, photo, handwriting) + shredders
  features/   deskew/normalise, edge reading, text-line features
  match/      pairwise edge costs (handcrafted + learned CNN)
  solve/      strips: greedy, beam, ATSP; grid.py: cross-cut; torn.py: torn paper
  verify/     Tesseract OCR readability, reranking, word lists (en, pcm, yo)
  cluster/    mixed-bin document separation
  render/     assembled image + confidence heatmap
  metrics.py  neighbour / direct accuracy, gap precision/recall
  pipeline.py prepare -> score -> solve -> Reconstruction
  cli.py      `reassemble synth | solve | bench`
tests/
```

## Credits & references

* DARPA Shredder Challenge (2011).
* D. Pomeranz, M. Shemesh, O. Ben-Shahar, "A fully automated greedy square jigsaw puzzle
  solver", CVPR 2011.
* A. Gallagher, "Jigsaw puzzles with pieces of unknown orientation", CVPR 2012.
* T. Paixão et al., deep-learning reconstruction of strip-shredded documents (2018–2020).
* Handwriting fonts: Caveat and Indie Flower (SIL Open Font License), from Fontsource.
