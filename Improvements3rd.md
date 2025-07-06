# Slitherlink Phase 3 – Loop Generation & JIT Overhead

> **Profile insight (after Phase 2.5):**
>
> - `generate_loop_with_symmetry → loop_wilson.generate → loop_builder.search_loop` dominates **≈ 13 s / 16 s**.
> - `random.shuffle` & `Random.getrandbits` called 10 M+ times.
> - Numba JIT compilation (`dispatcher.py`, `compiler.py`) costs **≈ 1 s** each run – repeated across boards.
>
> **Goal:** Cut loop‑generation cost by ≥ ×3 and make Numba compile once per session. Target 15×15 ≤ 8 s total.

---

## 1 – Task List (Phase 3)

| ID                                                               | Impact | Task                                                             | Implementation Hints                                                                          |
| ---------------------------------------------------------------- | ------ | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **L1**                                                           | ★★★★☆  | **Deterministic edge iteration – drop `random.shuffle` hotspot** | In `loop_builder.search_loop`, replace per‑step `random.shuffle(edges)` with:<br>\`\`\`python |
| edges = np.random.permutation(edge_index_array) # once per board |        |                                                                  |                                                                                               |

```<br>or reservoir sampling to avoid full shuffle each loop. |
| **L2** | ★★★★☆ | **Bitboard edge set & vectorised existence checks** | Store horizontal / vertical edge occupancy in two `uint8` NumPy arrays; use boolean masks instead of per‑edge `edge_exists()` Python calls. |
| **L3** | ★★★☆☆ | **Loop search rewritten as BFS over bitboards** | Break out of deep recursion; iterative queue with NumPy operations drastically reduces call count. |
| **L4** | ★★★☆☆ | **Wilson‑UST pure‑NumPy version** | Generate *uniform spanning tree* via Wilson using pre‑allocated arrays & vectorised random walks; union‑find in NumPy to cut Python loops. |
| **L5** | ★★★☆☆ | **Per‑process Numba warm‑up cache** | Call critical Numba‑compiled funcs once on worker spawn; or use `numba.caching` to write `.nbc` artefacts between runs. |
| **L6** | ★★☆☆☆ | **JIT compile guard** | Wrap compile‑heavy Numba functions with `@njit(cache=True)` and unit‑test at import time to avoid runtime compile hits. |
| **L7** | ★★☆☆☆ | **Adaptive max walk length** | Terminate random walk early if loop length exceeds 1.5 × board perimeter – avoids long wandering on large boards. |

---

## 2 – Milestones & Bench Targets

| Milestone | Expected Avg Time (15×15) | Notes |
|-----------|--------------------------|-------|
| After L1 & L2 | ~10 s | Shuffle & existence bottlenecks gone |
| After L3 & L4 | ≤ 6 s | Vectorised Wilson loop generation |
| After L5–L6   | compile cost ~0.1 s | First board slower okay; subsequent boards fast |

---

## 3 – Validation
1. `python -m timeit -s "from bench import run" "run(15,15,10)"`  → expect mean ≤ 6 s.
2. `python -Ximporttime -c "import generator"` to ensure Numba compile not repeated.
3. Profile again; top frame should now be ≤ 4 s for `loop_builder.search_loop`.

---

## 4 – Out‑of‑Scope (Phase 4)
* C1/C2 SAT variable reduction (still pending)
* Async generation farm
* GUI preview & manual tweaking

---

> **Implement L1 & L2 first** – they require no architectural change and give the biggest immediate win. Then proceed to vectorised rewrite (L3/L4) if further speed needed.

```
