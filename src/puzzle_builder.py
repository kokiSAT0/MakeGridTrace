"""パズル構築用のヘルパー関数をまとめたモジュール"""

from __future__ import annotations

# datetime モジュールから UTC 定数も合わせてインポート
from datetime import datetime, UTC
import math
import random
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from numba import njit

try:
    from . import sat_unique
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    import sat_unique

if TYPE_CHECKING:
    from src.solver import PuzzleSize, count_solutions
    from src.constants import _evaluate_difficulty
else:
    try:
        # パッケージとして実行された場合の相対インポート
        from .solver import PuzzleSize, count_solutions
        from .constants import _evaluate_difficulty
    except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
        # スクリプトとして直接実行されたときは同じディレクトリからインポートする
        from solver import PuzzleSize, count_solutions
        from constants import _evaluate_difficulty

Puzzle = Dict[str, Any]

# JSON スキーマのバージョン
SCHEMA_VERSION = "2.0"

# _evaluate_difficulty は constants モジュールに定義されている


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


@njit
def _quality_core(
    clues_arr: np.ndarray, curve_ratio: float, solver_steps: int, loop_length: int
) -> float:
    """Numba 対応の Quality Score 計算本体"""

    rows, cols = clues_arr.shape
    cells = rows * cols
    if cells == 0:
        return 0.0

    hint_count = 0
    for r in range(rows):
        for c in range(cols):
            if clues_arr[r, c] >= 0:
                hint_count += 1

    p = hint_count / cells
    if p == 0.0 or p == 1.0:
        entropy = 0.0
    else:
        entropy = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

    # 3x3 ブロックごとのヒント有無を調べ分散度を求める
    block_rows = (rows + 2) // 3
    block_cols = (cols + 2) // 3
    filled = 0
    for br in range(block_rows):
        for bc in range(block_cols):
            found = False
            for r in range(br * 3, min((br + 1) * 3, rows)):
                for c in range(bc * 3, min((bc + 1) * 3, cols)):
                    if clues_arr[r, c] >= 0:
                        found = True
                        break
                if found:
                    break
            if found:
                filled += 1
    dispersion = (
        filled / (block_rows * block_cols) if block_rows * block_cols > 0 else 0.0
    )

    density_score = 1.0 - min(1.0, abs(p - 0.25) * 4.0)

    if hint_count > 0:
        max_row = 0
        min_row = hint_count
        for r in range(rows):
            cnt = 0
            for c in range(cols):
                if clues_arr[r, c] >= 0:
                    cnt += 1
            if cnt > max_row:
                max_row = cnt
            if cnt < min_row:
                min_row = cnt
        max_col = 0
        min_col = hint_count
        for c in range(cols):
            cnt = 0
            for r in range(rows):
                if clues_arr[r, c] >= 0:
                    cnt += 1
            if cnt > max_col:
                max_col = cnt
            if cnt < min_col:
                min_col = cnt
        row_balance = 1.0 - (max_row - min_row) / hint_count
        col_balance = 1.0 - (max_col - min_col) / hint_count
    else:
        row_balance = 0.0
        col_balance = 0.0
    balance_score = max(0.0, (row_balance + col_balance) / 2.0)

    zero_pairs = 0
    for r in range(rows):
        for c in range(cols):
            if clues_arr[r, c] == 0:
                if r + 1 < rows and clues_arr[r + 1, c] == 0:
                    zero_pairs += 1
                if c + 1 < cols and clues_arr[r, c + 1] == 0:
                    zero_pairs += 1
    zero_ratio = zero_pairs / cells

    curve_score = 50.0 * min(1.0, max(0.0, (curve_ratio - 0.15) / 0.3))
    entropy_score = 50.0 * min(1.0, max(0.0, (entropy - 0.2) / 0.7))

    score = curve_score + entropy_score
    score += 15.0 * dispersion
    score += 10.0 * density_score
    score += 15.0 * balance_score
    score += min(10.0, 10000.0 / (solver_steps + 1))
    max_len = 2 * (rows + cols)
    length_ratio = min(1.0, loop_length / max_len) if max_len > 0 else 0.0
    score += 20.0 * length_ratio
    score -= 30.0 * zero_ratio

    if score < 0.0:
        score = 0.0
    if score > 100.0:
        score = 100.0
    return round(score, 2)


def _calculate_quality_score(
    clues: List[List[int | None]],
    curve_ratio: float,
    solver_steps: int,
    loop_length: int,
) -> float:
    """品質指標 (Quality Score) を計算する

    曲率比率に加え、ヒント密度や分散度など複数の統計から総合的な
    スコアを算出する関数。Phase 2 ではウェイト調整を行い、
    ``curve_ratio`` とエントロピーを最大 50 点まで、ソルバー手数を
    10 点、行列バランスを 15 点として評価する。

    ``loop_length`` が短すぎる場合は減点し、外周だけのループが
    高得点になるのを防ぐ。
    """

    arr = np.array(
        [[v if v is not None else -1 for v in row] for row in clues], dtype=np.int8
    )
    base = _quality_core(arr, curve_ratio, solver_steps, loop_length)
    return float(base)


def _calculate_edge_coverage(
    clues: List[List[int | None]], size: PuzzleSize
) -> Dict[tuple[int, int], int]:
    """各ヒントが単独でカバーする辺の数を計算するヘルパー

    周囲のセルにヒントがない辺のみをカウントする。境界上の辺は
    他セルと共有しないため、そのヒントの専有とみなす。
    """

    coverage: Dict[tuple[int, int], int] = {}
    for r in range(size.rows):
        for c in range(size.cols):
            if clues[r][c] is None:
                continue
            count = 0
            # 上辺
            if r == 0 or clues[r - 1][c] is None:
                count += 1
            # 下辺
            if r == size.rows - 1 or clues[r + 1][c] is None:
                count += 1
            # 左辺
            if c == 0 or clues[r][c - 1] is None:
                count += 1
            # 右辺
            if c == size.cols - 1 or clues[r][c + 1] is None:
                count += 1
            coverage[(r, c)] = count
    return coverage


def _reduce_clues(
    clues: List[List[int]],
    size: PuzzleSize,
    rng: random.Random,
    *,
    min_hint: int,
    step_limit: int | None = None,
) -> List[List[int | None]]:
    """ヒントを依存度の低い順に削減して一意性を保つ

    従来はランダム順で削除していたが、事前に各ヒントの "edge coverage"
    (そのヒントだけが参照する辺の数) を計算し、値が小さいものから
    順に試すことでソルバの呼び出し回数を減らす。

    以前は ``0`` が隣接するとエラーとしていたが、現在は ``0`` の隣接は
    少ないほど望ましいというソフト制約として扱うためチェックしない。

    :param rng: 乱数生成に利用する ``random.Random`` インスタンス
    :param step_limit: ソルバーに渡すステップ上限
    """

    result: List[List[int | None]] = [[v for v in row] for row in clues]

    # Edge coverage を計算し、値が小さいヒントから順に試す
    coverage = _calculate_edge_coverage(result, size)
    cells = [
        (coverage.get((r, c), 0), r, c)
        for r in range(size.rows)
        for c in range(size.cols)
        if result[r][c] is not None
    ]
    cells.sort()  # coverage が小さい順に並ぶ

    for _, r, c in cells:
        original = result[r][c]
        result[r][c] = None
        hint_count = sum(1 for row in result for v in row if v is not None)
        if hint_count < min_hint:
            result[r][c] = original
            continue

        # SAT ソルバーで一意解かどうか確認する
        if not sat_unique.is_unique(result, size):
            result[r][c] = original
            continue

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
    step_limit: int | None = None,
) -> List[List[int | None]]:
    """焼きなまし法でヒント配置を微調整する

    ヒントの追加・削除をランダムに行い、Quality Score が
    向上する場合に変更を採用する簡易アルゴリズムです。

    :param rng: 乱数生成に利用する ``random.Random`` インスタンス
    :param iterations: 試行回数。多いほど時間が掛かるが精度が上がる
    盤面が小さいときは温度を低め、大きいときは高めから始めて
    冷却率も緩やかにすることで過剰なランダム性を抑えます。
    """

    if step_limit is None:
        # 動的に探索上限を決める
        step_limit = size.rows * size.cols * 25

    best = [row[:] for row in clues]
    sols = count_solutions(best, size, limit=2, step_limit=step_limit)
    if sols != 1:
        return clues
    best_score = _calculate_quality_score(best, curve_ratio, solver_steps, loop_length)

    current = [row[:] for row in best]
    current_score = best_score

    # 盤面サイズに応じて初期温度と冷却率を調整する
    # セル数の平方根を基準スケールとし、小盤面は低温から、大盤面は高温から始める
    scale = max(0.5, math.sqrt(size.rows * size.cols) / 5)
    temperature = scale
    cooling_rate = 0.95 ** (1 / scale)

    for _ in range(iterations):
        # 冷却率を乗算しながら最低温度 0.01 を維持する
        temperature = max(0.01, temperature * cooling_rate)
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
    reason: str | None = None,
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
            "ruleVertex": solver_stats.get("rule_vertex", 0),
            "ruleClue": solver_stats.get("rule_clue", 0),
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
    if partial and reason is not None:
        puzzle["reason"] = reason
    return puzzle


__all__ = [
    "_build_puzzle_dict",
    "_reduce_clues",
    "_optimize_clues",
    "_calculate_quality_score",
    "_calculate_hint_dispersion",
]
