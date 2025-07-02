"""パズル構築用のヘルパー関数をまとめたモジュール"""

from __future__ import annotations

from datetime import datetime
import math
import random
from typing import Any, Dict, List, Optional

from .solver import PuzzleSize, count_solutions
from .constants import MAX_SOLVER_STEPS, _evaluate_difficulty

Puzzle = Dict[str, Any]

# JSON スキーマのバージョン
SCHEMA_VERSION = "2.0"

# MAX_SOLVER_STEPS と _evaluate_difficulty は constants モジュールに移動


def _calculate_quality_score(
    clues: List[List[int | None]], curve_ratio: float, solver_steps: int
) -> float:
    """曲率比率とヒント密度から簡易的な品質スコアを計算する"""

    cells = len(clues) * len(clues[0]) if clues else 0
    if cells == 0:
        return 0.0
    hint_count = sum(1 for row in clues for v in row if v is not None)
    p = hint_count / cells
    if p in (0.0, 1.0):
        entropy = 0.0
    else:
        entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    score = 20 * math.log10(max(curve_ratio * 100, 0.01)) + 25 * entropy
    score += min(20.0, 10000.0 / (solver_steps + 1))
    return round(max(0.0, min(100.0, score)), 2)


def _reduce_clues(
    clues: List[List[int]],
    size: PuzzleSize,
    rng: random.Random,
    *,
    min_hint: int,
) -> List[List[int | None]]:
    """ヒントをランダムに削減して一意性を保つ

    :param rng: 乱数生成に利用する ``random.Random`` インスタンス
    """

    result: List[List[int | None]] = [[v for v in row] for row in clues]
    cells = [(r, c) for r in range(size.rows) for c in range(size.cols)]
    rng.shuffle(cells)

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
    theme: Optional[str],
    generation_params: Dict[str, Any],
    seed_hash: str,
) -> Puzzle:
    """パズル用の辞書オブジェクトを構築するヘルパー関数"""

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    qs = _calculate_quality_score(clues, curve_ratio, solver_stats["steps"])
    puzzle: Puzzle = {
        "schemaVersion": SCHEMA_VERSION,
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
        "theme": theme,
        "generationParams": generation_params,
        "seedHash": seed_hash,
        "qualityScore": qs,
        "createdBy": "auto-gen-v1",
        "createdAt": datetime.utcnow().date().isoformat(),
    }
    return puzzle


__all__ = ["_build_puzzle_dict", "_reduce_clues", "_calculate_quality_score"]
