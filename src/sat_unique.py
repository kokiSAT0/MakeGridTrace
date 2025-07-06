"""PySAT を使った一意解チェックモジュール"""

from __future__ import annotations

from typing import List
import itertools

from pysat.formula import CNF, IDPool

# EncType は PySAT で定義されている列挙型で、
# エンコーディング方式を数値で表現します
from pysat.card import CardEnc, EncType
from pysat.solvers import Minisat22

try:
    from .solver import PuzzleSize
except ImportError:  # pragma: no cover
    from solver import PuzzleSize


def _create_variables(
    size: PuzzleSize, pool: IDPool
) -> tuple[List[List[int]], List[List[int]]]:
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


def _even_parity_clauses(lits: List[int]) -> List[List[int]]:
    """偶数個の真を強制する制約を生成"""
    clauses: List[List[int]] = []
    for bits in itertools.product([0, 1], repeat=len(lits)):
        # 奇数個のときはその割り当てを禁止する
        if sum(bits) % 2 == 1:
            clause = []
            for lit, bit in zip(lits, bits):
                clause.append(-lit if bit else lit)
            clauses.append(clause)
    return clauses


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
                # EncType.seqcounter を指定して 2 本以下に制限する
                cnf.extend(
                    CardEnc.atmost(
                        lits,
                        2,
                        vpool=pool,
                        encoding=EncType.seqcounter,
                    ).clauses
                )
                # from_xor が利用できない環境のため、自前で偶数パリティを追加
                cnf.extend(_even_parity_clauses(lits))
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
            cnf.extend(
                CardEnc.equals(
                    lits,
                    clue,
                    vpool=pool,
                    encoding=EncType.seqcounter,
                ).clauses
            )

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
