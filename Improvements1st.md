# Slitherlink Map Generator – Phase 1: Speed‑First Improvements

> **Goal:** Within one sprint (≈ 1 week) cut the _average_ generation time by **≥ 10×** on 10×10–20×20 boards **while still returning puzzles that meet**:
>
> - Quality Score ≥ 50
> - Unique solution

## 0. Baseline & Acceptance

| Metric                           | Current | Target     | Measured by                        |
| -------------------------------- | ------- | ---------- | ---------------------------------- |
| 10 × 10 board avg.               | *N* s   | **≤ 1 s**  | `bench/bench.py` (timeit 100 runs) |
| 20 × 20 board avg.               | *M* s   | **≤ 15 s** | same as above                      |
| Fail‑rate (timeout / non‑unique) | ≥ X %   | **≤ 5 %**  | CI summary                         |

Add a `bench` folder with the script below and wire it into GitHub Actions so every PR gets a before/after report.

```bash
python -m timeit -s "from src.generator import generate_puzzle" \
  "generate_puzzle(rows={rows}, cols={cols}, difficulty='normal')"
```

---

## 1. Task List (do in order)

| ID                                                                     | Impact | Task                                                | Hints                                                              |
| ---------------------------------------------------------------------- | ------ | --------------------------------------------------- | ------------------------------------------------------------------ |
| **A1**                                                                 | ★★★★☆  | **Replace DFS loop generator with Wilson‑UST**      | • New `generator/loop_wilson.py` returns a loop in ≤ O(N M)        |
| • Convert spanning tree to loop by connecting leaf to root             |        |                                                     |                                                                    |
| • Preserve `symmetry` by post‑processing                               |        |                                                     |                                                                    |
|                                                                        |        |                                                     |                                                                    |
| **A2**                                                                 | ★★★★☆  | **Unique‑solution check via SAT (PySAT + Minisat)** | • Encode current board to CNF with “≤1 solution” constraint        |
| • Run `Solver.solve_limited(expect_interrupt=True)`; stop on 2nd model |        |                                                     |                                                                    |
| • Drop‐in replace `count_solutions` (keep API)                         |        |                                                     |                                                                    |
|                                                                        |        |                                                     |                                                                    |
| **A3**                                                                 | ★★★☆☆  | **Dynamic `solver_step_limit`**                     | • Set to `rows * cols * 25`                                        |
| • Remove hard‑coded 500 000                                            |        |                                                     |                                                                    |
|                                                                        |        |                                                     |                                                                    |
| **A4**                                                                 | ★★★☆☆  | **Profiling & stats pipeline**                      | • Add `--profile` flag that writes `stats.prof`                    |
| • Document use of `snakeviz`                                           |        |                                                     |                                                                    |
|                                                                        |        |                                                     |                                                                    |
| **A5**                                                                 | ★★☆☆☆  | **Timeout reason logging**                          | • When `timeout_s` exceeded, add `reason: "timeout"` field in JSON |
|                                                                        |        |                                                     |                                                                    |

> **Milestone complete when:** All benchmarks meet targets **and** CI green.

---

## 2. Implementation Notes

### A1 – Wilson Algorithm Sketch

1. Choose random root vertex.
2. For every unvisited vertex, perform loop‑erased random walk until hitting visited set.
3. Union all edges → spanning tree.
4. Convert to loop: find any leaf, walk to root, then add edge from leaf→root.
5. For themes (`maze`, `spiral`…), perturb edge weights before walk to bias shape.

### A2 – SAT Encoding Tips

- Variables: one per edge ⇒ boolean "edge in loop".
- Degree‑2 constraint per vertex: 0, 2 → encode via pairwise ⊕ clauses.
- "At‑most‑1 additional solution" trick: after first model, add blocking clause and solve again with _conflict limit = 1 000_.

### Quality Guard

Keep the existing Quality Score calculation **unchanged** for Phase 1; we will revisit weighting in Phase 2.

---

## 3. Deliverables

1. **PR #phase1‑speed** containing:

   - New modules `loop_wilson.py`, `sat_unique.py`
   - Updated `generator.py` integration
   - Bench & CI workflow

2. **Bench results** posted in PR description.
3. **Short hindsight doc** (markdown) listing what actually moved the needle.

---

## 4. Out‑of‑Scope (Phase 2+)

- NumPy/Numba bitboard refactor
- Clue reduction heuristics rewrite
- Theme & Quality Score expansion
- Full parallel generation farm

---

### Quick Start for Programmers

```bash
pip install numpy pysat snakeviz
pytest  # ensure no regressions
python bench/bench.py
```
