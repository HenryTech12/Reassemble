"""Solvers that turn a pairwise cost matrix into a left-to-right strip order.

Ordering strips is a *Hamiltonian path* problem: visit every piece once, minimising the
sum of seam costs.  Adding one "virtual boundary" node (index ``n``) whose costs are the
start/end priors turns it into an **asymmetric travelling-salesman tour**: the tour goes
boundary -> first strip -> ... -> last strip -> boundary.  Three solvers are provided:

* ``greedy``: best-buddy chaining.  Fast and surprisingly strong.
* ``beam``: left-to-right beam search; also yields the top-K alternatives that the OCR
  reranker (Phase 2) chooses between.
* ``atsp``: local search (Or-opt + 2-opt) on the ATSP, seeded from the other two.

User constraints (locked joins from the editor) are compiled into the cost matrix, so all
three solvers respect them for free.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

BIG = 1e6
LOCK = -1e3


@dataclass
class SolveResult:
    order: list[int]                 # piece indices left to right
    cost: float
    method: str
    runtime_s: float = 0.0
    alternatives: list[tuple[float, list[int]]] = field(default_factory=list)


def build_atsp(C: np.ndarray, start: np.ndarray | None = None, end: np.ndarray | None = None,
               locked: list[tuple[int, int]] | None = None,
               forbidden: list[tuple[int, int]] | None = None) -> np.ndarray:
    """(n+1)x(n+1) ATSP matrix; node n is the page boundary.

    ``locked`` pairs (a, b) force b directly right of a: every other successor of a and
    every other predecessor of b becomes prohibitively expensive.  ``forbidden`` pairs
    (a user "split") get a huge cost.
    """
    n = len(C)
    M = np.full((n + 1, n + 1), BIG, dtype=np.float64)
    M[:n, :n] = np.minimum(C, BIG)
    M[n, :n] = 0.0 if start is None else start
    M[:n, n] = 0.0 if end is None else end
    np.fill_diagonal(M, BIG)
    for a, b in forbidden or []:
        M[a, b] = BIG
    for a, b in locked or []:
        M[a, :] = BIG
        M[:, b] = BIG
        M[a, b] = LOCK
    return M


def tour_cost(M: np.ndarray, tour: list[int] | np.ndarray) -> float:
    t = np.asarray(tour)
    return float(M[t, np.roll(t, -1)].sum())


def tour_to_order(tour: list[int], n: int) -> list[int]:
    k = tour.index(n)
    t = tour[k + 1:] + tour[:k]
    return t


def order_to_tour(order: list[int], n: int) -> list[int]:
    return [n] + list(order)


# --------------------------------------------------------------------------- greedy


def greedy_chain(M: np.ndarray) -> list[int]:
    """Best-buddy greedy chaining on the ATSP matrix (boundary node excluded).

    1. Link *best buddies* first: pairs where b is a's best successor **and** a is b's
       best predecessor.  Mutual agreement is strong evidence (Pomeranz et al.).
    2. Then add remaining pairs cheapest-first, as long as every piece keeps at most one
       successor/predecessor and no cycle forms (union-find, like Kruskal's algorithm).
    3. The resulting chains are concatenated, again greedily, using the boundary costs.
    """
    n = len(M) - 1
    C = M[:n, :n]
    succ = -np.ones(n, int)
    pred = -np.ones(n, int)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def link(a, b):
        if a == b or succ[a] >= 0 or pred[b] >= 0 or C[a, b] >= BIG:
            return False
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        succ[a], pred[b] = b, a
        parent[ra] = rb
        return True

    best_s = C.argmin(1)
    best_p = C.argmin(0)
    buddies = [(C[a, best_s[a]], a, best_s[a]) for a in range(n) if best_p[best_s[a]] == a]
    for _, a, b in sorted(buddies):
        link(a, int(b))
    flat = np.argsort(C, axis=None)
    for idx in flat:
        a, b = divmod(int(idx), n)
        if C[a, b] >= BIG:
            break
        link(a, b)
    # collect chains
    chains = []
    for s in range(n):
        if pred[s] < 0:
            ch = [s]
            while succ[ch[-1]] >= 0:
                ch.append(int(succ[ch[-1]]))
            chains.append(ch)
    # order chains: start with the chain whose head is the best page start, then append
    # the chain whose head best continues the current tail
    chains.sort(key=lambda ch: M[n, ch[0]])
    out = chains.pop(0)
    while chains:
        k = min(range(len(chains)), key=lambda i: M[out[-1], chains[i][0]])
        out += chains.pop(k)
    return out


# --------------------------------------------------------------------------- beam search


def beam_search(M: np.ndarray, width: int = 200, top_k: int = 10) -> list[tuple[float, list[int]]]:
    """Left-to-right beam search over orderings.

    A state is a partial sequence; we extend every state in the beam by every unused
    piece and keep the ``width`` cheapest.  States that used the same set of pieces and end
    on the same piece are interchangeable for the future, so only the cheapest is kept
    (this is the dynamic-programming insight of Held–Karp, applied to the beam).
    Returns up to ``top_k`` complete sequences sorted by total cost.
    """
    n = len(M) - 1
    beam = [(0.0, (n,), 0)]  # cost, sequence (starts at boundary), used bitmask
    for _ in range(n):
        cand = {}
        for cost, seq, used in beam:
            last = seq[-1]
            row = M[last, :n]
            for j in np.argsort(row)[: min(n, 3 * width)]:
                j = int(j)
                if used >> j & 1 or row[j] >= BIG:
                    continue
                c = cost + row[j]
                key = (used | (1 << j), j)
                if key not in cand or cand[key][0] > c:
                    cand[key] = (c, seq + (j,), used | (1 << j))
        if not cand:
            break
        beam = sorted(cand.values(), key=lambda s: s[0])[:width]
    done = []
    for cost, seq, used in beam:
        if len(seq) == n + 1:
            done.append((cost + M[seq[-1], n], list(seq[1:])))
    done.sort(key=lambda s: s[0])
    return done[:top_k]


# --------------------------------------------------------------------------- ATSP local search


def or_opt_2opt(M: np.ndarray, tour: list[int], max_rounds: int = 500) -> list[int]:
    """Improve an ATSP tour by local moves until no move helps.

    * **Or-opt**: cut out a segment of 1-3 consecutive nodes and reinsert it elsewhere
      (direction preserved).  This fixes a strip or a small group placed in the wrong spot.
    * **2-opt**: reverse a segment.  On an *asymmetric* problem reversing changes the cost of
      every edge inside the segment, so we use prefix sums of forward and backward edge costs
      to evaluate each reversal in O(1).

    Every candidate move is scored by its *delta* (cost change) with NumPy, so a full
    neighbourhood scan is cheap; we apply the best improving move each round.  This is
    the core of Lin–Kernighan-style heuristics (LKH), stripped to the moves that matter.
    """
    t = np.array(tour)
    n = len(t)
    if n < 4:
        return list(t)
    for _ in range(max_rounds):
        best_delta, best_move = -1e-9, None
        # ---- Or-opt
        for L in (1, 2, 3):
            if L > n - 3:
                break
            for i in range(n):
                seg = np.roll(t, -i)[:L]
                rest = np.roll(t, -i)[L:]  # starts at q (node after seg), ends at p
                p, q = rest[-1], rest[0]
                gain = M[p, seg[0]] + M[seg[-1], q] - M[p, q]
                u, v = rest[:-1], rest[1:]  # insertion edges (u,v), excluding (p,q)
                add = M[u, seg[0]] + M[seg[-1], v] - M[u, v]
                k = int(np.argmin(add))
                d = add[k] - gain
                if d < best_delta:
                    best_delta, best_move = d, ("or", i, L, k)
        # ---- 2-opt (asymmetric): reverse t[i..j]
        fwd = M[t, np.roll(t, -1)]           # fwd[k] = cost t_k -> t_{k+1}
        bwd = M[np.roll(t, -1), t]           # bwd[k] = cost t_{k+1} -> t_k
        cf = np.concatenate([[0], np.cumsum(fwd)])
        cb = np.concatenate([[0], np.cumsum(bwd)])
        ii, jj = np.triu_indices(n - 1, 1)
        ii, jj = ii + 1, jj + 1               # 1 <= i < j <= n-1 (keep t[0] fixed)
        jn = (jj + 1) % n
        d2 = (M[t[ii - 1], t[jj]] + M[t[ii], t[jn]] - M[t[ii - 1], t[ii]] - M[t[jj], t[jn]]
              + (cb[jj] - cb[ii]) - (cf[jj] - cf[ii]))
        if len(d2):
            k = int(np.argmin(d2))
            if d2[k] < best_delta:
                best_delta, best_move = d2[k], ("2opt", int(ii[k]), int(jj[k]))
        if best_move is None:
            break
        if best_move[0] == "or":
            _, i, L, k = best_move
            r = np.roll(t, -i)
            seg, rest = r[:L], r[L:]
            t = np.concatenate([rest[:k + 1], seg, rest[k + 1:]])
        else:
            _, i, j = best_move
            t = np.concatenate([t[:i], t[i:j + 1][::-1], t[j + 1:]])
    return [int(x) for x in t]


def join_min_marginals(M: np.ndarray, tour: list[int]) -> list[float]:
    """For every edge of ``tour``: how much the best tour cost rises if that edge is banned.

    This is a far better confidence signal than comparing a pair with its alternatives in
    isolation: a join can look only marginally better than other candidates locally, yet
    be *forced* by the rest of the solution (every other neighbour is already taken).
    We approximate the re-solve with the local search above, warm-started from the
    current tour.
    """
    base = tour_cost(M, tour)
    out = []
    for k in range(len(tour)):
        a, b = tour[k], tour[(k + 1) % len(tour)]
        M2 = M.copy()
        M2[a, b] = BIG
        t2 = or_opt_2opt(M2, tour)
        out.append(max(0.0, tour_cost(M2, t2) - base))
    return out


# --------------------------------------------------------------------------- driver


def solve_strips(C: np.ndarray, start: np.ndarray | None = None, end: np.ndarray | None = None,
                 locked=None, forbidden=None, methods=("greedy", "beam", "atsp"),
                 beam_width: int = 200, top_k: int = 10) -> dict[str, SolveResult]:
    """Run the requested solvers; returns {method: SolveResult} plus ``best``."""
    n = len(C)
    M = build_atsp(C, start, end, locked, forbidden)
    results: dict[str, SolveResult] = {}
    if n == 1:
        r = SolveResult([0], 0.0, "trivial")
        return {"best": r, "trivial": r}
    if "greedy" in methods or "atsp" in methods:
        t0 = time.perf_counter()
        g = greedy_chain(M)
        results["greedy"] = SolveResult(g, tour_cost(M, order_to_tour(g, n)), "greedy",
                                        time.perf_counter() - t0)
    if "beam" in methods or "atsp" in methods:
        t0 = time.perf_counter()
        alts = beam_search(M, beam_width, top_k)
        if alts:
            results["beam"] = SolveResult(alts[0][1], alts[0][0], "beam",
                                          time.perf_counter() - t0, alternatives=alts)
    if "atsp" in methods:
        t0 = time.perf_counter()
        best_t, best_c = None, np.inf
        seeds = [r.order for r in results.values()]
        for s in seeds:
            t = or_opt_2opt(M, order_to_tour(s, n))
            c = tour_cost(M, t)
            if c < best_c:
                best_t, best_c = t, c
        results["atsp"] = SolveResult(tour_to_order(best_t, n), best_c, "atsp",
                                      time.perf_counter() - t0)
    chosen = {k: v for k, v in results.items() if k in methods}
    best = min(chosen.values(), key=lambda r: r.cost)
    results["best"] = best
    return results
