"""PySAT を使った一意解チェックモジュール"""

from __future__ import annotations

from typing import List

from pysat.formula import CNF, IDPool
from pysat.card import CardEnc
from pysat.solvers import Minisat22

try:
    from .solver import PuzzleSize
except ImportError:  # pragma: no cover
    from solver import PuzzleSize


def _create_variables(size: PuzzleSize, pool: IDPool) -> tuple[List[List[int]], List[List[int]]]:
    """辺ごとの SAT 変数を作成する補助関数"""
    horizontal: List[List[int]] = []
    for r in range(size.rows + 1):
        row = []
        for c in range(size.cols):
            row.append(pool.id(f"h_{r}_{c}"))
        horizontal.append(row)

    vertical: List[List[int]] = []
    for r in range(size.rows):
        row = []
        for c in range(size.cols + 1):
            row.append(pool.id(f"v_{r}_{c}"))
        vertical.append(row)
    return horizontal, vertical


def is_unique(clues: List[List[int | None]], size: PuzzleSize) -> bool:
    """与えられたヒントから解が一意か確認する"""
    pool = IDPool()
    horizontal, vertical = _create_variables(size, pool)

    cnf = CNF()

    # 頂点の次数は 0 または 2 に制限する
    for r in range(size.rows + 1):
        for c in range(size.cols + 1):
            lits: List[int] = []
            if c < size.cols:
                lits.append(horizontal[r][c])
            if c > 0:
                lits.append(horizontal[r][c - 1])
            if r < size.rows:
                lits.append(vertical[r][c])
            if r > 0:
                lits.append(vertical[r - 1][c])
            if len(lits) >= 2:
                cnf.extend(CardEnc.atmost(lits, 2, vpool=pool, encoding="pairwise").clauses)
                cnf.extend(CNF.from_xor(lits, rhs=False, vpool=pool).clauses)
            elif len(lits) == 1:
                cnf.append([-lits[0]])

    # ヒント数字の制約を追加
    for r in range(size.rows):
        for c in range(size.cols):
            clue = clues[r][c]
            if clue is None:
                continue
            lits = [
                horizontal[r][c],
                horizontal[r + 1][c],
                vertical[r][c],
                vertical[r][c + 1],
            ]
            cnf.extend(CardEnc.equals(lits, clue, vpool=pool, encoding="seqcounter").clauses)

    with Minisat22(bootstrap_with=cnf.clauses) as solver:
        if not solver.solve():
            return False
        model = solver.get_model()
        blocking = []
        for var in range(1, pool.top + 1):
            if model[var - 1] > 0:
                blocking.append(-var)
            else:
                blocking.append(var)
        solver.add_clause(blocking)
        unique = not solver.solve()
    return unique
