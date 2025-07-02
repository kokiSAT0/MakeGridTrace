"""パズル生成補助関数をまとめたモジュール"""

from __future__ import annotations

from datetime import datetime
import random
from typing import Any, Dict, List, Optional

from .solver import PuzzleSize, count_solutions

# generator との循環参照を避けるため型エイリアスをここでも定義
Puzzle = Dict[str, Any]

MAX_SOLVER_STEPS = 500000


def _reduce_clues(
    clues: List[List[int]], size: PuzzleSize, *, min_hint: int
) -> List[List[int | None]]:
    """ヒントをランダムに削減して一意性を保つ"""
    result: List[List[int | None]] = [[v for v in row] for row in clues]
    cells = [(r, c) for r in range(size.rows) for c in range(size.cols)]
    random.shuffle(cells)

    for r, c in cells:
        if result[r][c] is None:
            continue
        original = result[r][c]
        result[r][c] = None
        hint_count = sum(1 for row in result for v in row if v is not None)
        if (
            hint_count < min_hint
            or count_solutions(result, size, limit=2, step_limit=MAX_SOLVER_STEPS) != 1
        ):
            result[r][c] = original

    return result


def _build_puzzle_dict(
    *,
    size: PuzzleSize,
    edges: Dict[str, List[List[bool]]],
    clues: List[List[int | None]],
    clues_full: List[List[int]],
    loop_length: int,
    curve_ratio: float,
    difficulty: str,
    solver_stats: Dict[str, int],
    symmetry: Optional[str],
    generation_params: Dict[str, Any],
    seed_hash: str,
) -> Puzzle:
    """パズル用の辞書オブジェクトを構築するヘルパー関数"""

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    puzzle: Puzzle = {
        "schemaVersion": "2.0",
        "id": f"sl_{size.rows}x{size.cols}_{difficulty}_{timestamp}",
        "size": {"rows": size.rows, "cols": size.cols},
        "clues": clues,
        "cluesFull": clues_full,
        "solutionEdges": edges,
        "loopStats": {"length": loop_length, "curveRatio": curve_ratio},
        "solverStats": {
            "steps": solver_stats["steps"],
            "maxDepth": solver_stats["max_depth"],
        },
        "difficulty": difficulty,
        "difficultyEval": _evaluate_difficulty(
            solver_stats["steps"], solver_stats["max_depth"]
        ),
        "symmetry": symmetry,
        "generationParams": generation_params,
        "seedHash": seed_hash,
        "createdBy": "auto-gen-v1",
        "createdAt": datetime.utcnow().date().isoformat(),
    }
    return puzzle


def _evaluate_difficulty(steps: int, depth: int) -> str:
    """ソルバー統計から難易度を推定する関数"""

    if steps < 1000 and depth <= 2:
        return "easy"
    if steps < 10000 and depth <= 10:
        return "normal"
    if steps < 100000 and depth <= 30:
        return "hard"
    return "expert"


__all__ = ["_reduce_clues", "_build_puzzle_dict", "_evaluate_difficulty"]
