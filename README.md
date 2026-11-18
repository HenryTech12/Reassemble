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
reassemble bench    # accuracy benchmark on fresh synthetic pages
```

`solve` writes `reconstruction.png`, `heatmap.png` (joins coloured by confidence) and
`result.json`.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Synthetic data generator + digital shredders | ✅ |
| 1 | Strip reconstruction MVP | ✅ |
| 2–8 | OCR, two-sided, mixed bins, learned matcher, cross-cut, capture, web app, ... | in progress |

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

## Project layout

```
reassemble/
  synth/      page renderer (memo, letter, invoice, photo, handwriting) + shredders
  features/   deskew/normalise, edge reading, text-line features
  match/      pairwise edge costs
  solve/      greedy, beam search, ATSP local search, gap detection
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
