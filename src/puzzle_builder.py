"""パズル構築用のヘルパー関数をまとめたモジュール"""

from __future__ import annotations

# datetime モジュールから UTC 定数も合わせてインポート
from datetime import datetime, UTC
import math
import random
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.solver import PuzzleSize, count_solutions
    from src.constants import MAX_SOLVER_STEPS, _evaluate_difficulty
    from src import validator
else:
    try:
        # パッケージとして実行された場合の相対インポート
        from .solver import PuzzleSize, count_solutions
        from .constants import MAX_SOLVER_STEPS, _evaluate_difficulty
        from . import validator
    except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
        # スクリプトとして直接実行されたときは同じディレクトリからインポートする
        from solver import PuzzleSize, count_solutions
        from constants import MAX_SOLVER_STEPS, _evaluate_difficulty
        import validator

Puzzle = Dict[str, Any]

# JSON スキーマのバージョン
SCHEMA_VERSION = "2.0"

# MAX_SOLVER_STEPS と _evaluate_difficulty は constants モジュールに移動


def _calculate_hint_dispersion(clues: List[List[int | None]]) -> float:
    """ヒントが盤面に均等に散らばっている度合いを返す

    盤面を 3x3 のブロックに分け、各ブロックに少なくとも1つヒントが存在する
    割合を計算することでヒントの偏りを数値化する。
    """

    rows = len(clues)
    cols = len(clues[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        return 0.0

    block_rows = (rows + 2) // 3
    block_cols = (cols + 2) // 3
    total = block_rows * block_cols
    filled = 0

    for br in range(block_rows):
        for bc in range(block_cols):
            found = False
            for r in range(br * 3, min((br + 1) * 3, rows)):
                for c in range(bc * 3, min((bc + 1) * 3, cols)):
                    if clues[r][c] is not None:
                        found = True
                        break
                if found:
                    break
            if found:
                filled += 1

    return filled / total if total else 0.0


def _calculate_quality_score(
    clues: List[List[int | None]],
    curve_ratio: float,
    solver_steps: int,
    loop_length: int,
) -> float:
    """曲率比率とヒントの配置から品質スコアを計算する

    ``loop_length`` の長さも評価に取り入れ、外周をなぞるだけの短い
    ループが高得点にならないよう調整する。
    """

    cells = len(clues) * len(clues[0]) if clues else 0
    if cells == 0:
        return 0.0
    hint_count = sum(1 for row in clues for v in row if v is not None)
    p = hint_count / cells
    if p in (0.0, 1.0):
        entropy = 0.0
    else:
        entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    dispersion = _calculate_hint_dispersion(clues)

    score = 20 * math.log10(max(curve_ratio * 100, 0.01)) + 25 * entropy
    score += 15 * dispersion
    score += min(20.0, 10000.0 / (solver_steps + 1))
    # 盤面サイズに対するループ長の割合を 0~1 で計算しスコアに加算
    max_len = 2 * (len(clues) + len(clues[0]))
    length_ratio = min(1.0, loop_length / max_len) if max_len else 0.0
    score += 20 * length_ratio
    return round(max(0.0, min(100.0, score)), 2)


def _reduce_clues(
    clues: List[List[int]],
    size: PuzzleSize,
    rng: random.Random,
    *,
    min_hint: int,
    step_limit: int = MAX_SOLVER_STEPS,
) -> List[List[int | None]]:
    """ヒントをランダムに削減して一意性を保つ

    生成途中で ``0`` が縦横に隣接してしまうとハード制約 H-8 を満たさなくなる。
    そのため削除後に ``_has_zero_adjacent`` を確認し、違反する場合は削除を戻す。

    :param rng: 乱数生成に利用する ``random.Random`` インスタンス
    :param step_limit: ソルバーに渡すステップ上限
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
            or count_solutions(result, size, limit=2, step_limit=step_limit) != 1
            or validator._has_zero_adjacent(
                [[v if v is not None else -1 for v in row] for row in result]
            )
        ):
            result[r][c] = original

    return result


def _optimize_clues(
    clues: List[List[int | None]],
    clues_full: List[List[int]],
    size: PuzzleSize,
    rng: random.Random,
    *,
    min_hint: int,
    loop_length: int,
    curve_ratio: float,
    solver_steps: int,
    iterations: int = 30,
    step_limit: int = MAX_SOLVER_STEPS,
) -> List[List[int | None]]:
    """焼きなまし法でヒント配置を微調整する

    ヒントの追加・削除をランダムに行い、Quality Score が
    向上する場合に変更を採用する簡易アルゴリズムです。

    :param rng: 乱数生成に利用する ``random.Random`` インスタンス
    :param iterations: 試行回数。多いほど時間が掛かるが精度が上がる
    """

    best = [row[:] for row in clues]
    sols = count_solutions(best, size, limit=2, step_limit=step_limit)
    if sols != 1:
        return clues
    best_score = _calculate_quality_score(best, curve_ratio, solver_steps, loop_length)

    current = [row[:] for row in best]
    current_score = best_score
    temperature = 1.0

    for _ in range(iterations):
        temperature = max(0.01, temperature * 0.95)
        r = rng.randrange(size.rows)
        c = rng.randrange(size.cols)
        candidate = [row[:] for row in current]
        if candidate[r][c] is None:
            candidate[r][c] = clues_full[r][c]
        else:
            hint_count = sum(1 for row in candidate for v in row if v is not None)
            if hint_count <= min_hint:
                continue
            candidate[r][c] = None

        if validator._has_zero_adjacent(
            [[v if v is not None else -1 for v in row] for row in candidate]
        ):
            continue

        if count_solutions(candidate, size, limit=2, step_limit=step_limit) != 1:
            continue

        cand_score = _calculate_quality_score(
            candidate, curve_ratio, solver_steps, loop_length
        )
        delta = cand_score - current_score
        if delta > 0 or math.exp(delta / temperature) > rng.random():
            current = candidate
            current_score = cand_score
            if cand_score > best_score:
                best = candidate
                best_score = cand_score

    return best


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
    partial: bool = False,
) -> Puzzle:
    """パズル用の辞書オブジェクトを構築するヘルパー関数"""

    # timezone-aware な UTC 時刻を取得する
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    qs = _calculate_quality_score(
        clues, curve_ratio, solver_stats["steps"], loop_length
    )
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
        # ISO8601 形式の UTC 日付文字列を保存
        "createdAt": datetime.now(UTC).date().isoformat(),
        "partial": partial,
    }
    return puzzle


__all__ = [
    "_build_puzzle_dict",
    "_reduce_clues",
    "_optimize_clues",
    "_calculate_quality_score",
    "_calculate_hint_dispersion",
]
