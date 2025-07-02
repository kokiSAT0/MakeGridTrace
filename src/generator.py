"""簡易的なスリザーリンク盤面生成モジュール"""

from __future__ import annotations

import logging
import time
import os
import random
import concurrent.futures
import hashlib
from typing import Any, Dict, List, Optional, cast, TYPE_CHECKING

if TYPE_CHECKING:
    from src.solver import PuzzleSize, calculate_clues, count_solutions
else:
    try:
        # パッケージとして実行された場合の相対インポート
        from .solver import PuzzleSize, calculate_clues, count_solutions
    except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
        # スクリプトとして直接実行されたときは同じディレクトリからインポートする
        from solver import (
            PuzzleSize,
            calculate_clues,
            count_solutions,
        )

from .loop_builder import (
    _create_empty_edges,
    _generate_random_loop,
    _count_edges,
    _calculate_curve_ratio,
    _apply_rotational_symmetry,
)
from .puzzle_io import save_puzzle
from .validator import validate_puzzle, _has_zero_adjacent
from .puzzle_builder import _build_puzzle_dict, _reduce_clues


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


Puzzle = Dict[str, Any]

# JSON スキーマのバージョン
SCHEMA_VERSION = "2.0"


ALLOWED_DIFFICULTIES = {"easy", "normal", "hard", "expert"}


# difficulty ごとの最小ヒント比率。盤面のセル数に掛けて下限ヒント数を求める
MIN_HINT_RATIO = {
    "easy": 0.3,
    "normal": 0.2,
    "hard": 0.15,
    "expert": 0.1,
}

# 生成失敗時に何回まで再試行するか
# リトライ回数を増やしてヒント削減に失敗しにくくする
RETRY_LIMIT = 5

# ソルバーが探索する最大ステップ数。超えると途中で打ち切る
# ソルバーのステップ上限を増加させて解の探索精度を向上させる
MAX_SOLVER_STEPS = 500000


def _evaluate_difficulty(steps: int, depth: int) -> str:
    """ソルバー統計から難易度を推定する関数"""

    # 解析ステップ数とバックトラック深さを基準に難易度を決める
    if steps < 1000 and depth <= 2:
        return "easy"
    if steps < 10000 and depth <= 10:
        return "normal"
    if steps < 100000 and depth <= 30:
        return "hard"
    return "expert"


def generate_puzzle(
    rows: int,
    cols: int,
    difficulty: str = "normal",
    *,
    seed: int | None = None,
    symmetry: Optional[str] = None,
    return_stats: bool = False,
) -> Puzzle | tuple[Puzzle, Dict[str, int]]:
    """簡易な盤面を生成して返す

    :param rows: 盤面の行数
    :param cols: 盤面の列数
    :param difficulty: 難易度ラベル
    :param seed: 乱数シード。再現したいときに指定する
    :param symmetry: 回転対称を指定する場合は "rotational"
    :param return_stats: True なら生成統計も返す
    :return: 生成したパズル。``return_stats`` が True の場合は
        ``(Puzzle, dict)`` のタプルを返す

    生成結果の ``solverStats`` フィールドには、
    ソルバーが使った探索ステップ数 (``steps``) と
    最大探索深さ (``maxDepth``) を記録します。
    """

    if difficulty not in ALLOWED_DIFFICULTIES:
        raise ValueError(f"difficulty は {ALLOWED_DIFFICULTIES} のいずれかで指定")

    if seed is not None:
        random.seed(seed)

    generation_params = {
        "rows": rows,
        "cols": cols,
        "difficulty": difficulty,
        "seed": seed,
        "symmetry": symmetry,
    }

    seed_hash = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()

    start_time = time.perf_counter()
    logger.info("盤面生成開始: %dx%d difficulty=%s", rows, cols, difficulty)

    size = PuzzleSize(rows=rows, cols=cols)

    last_edges: Dict[str, List[List[bool]]] | None = None
    for attempt in range(RETRY_LIMIT):
        step_time = time.perf_counter()
        edges = _create_empty_edges(size)
        logger.info("空の盤面作成: %.3f 秒", time.perf_counter() - step_time)

        step_time = time.perf_counter()
        _generate_random_loop(edges, size)
        if symmetry == "rotational":
            # 回転対称を希望する場合は生成したループを180度回転して重ねる
            _apply_rotational_symmetry(edges, size)
        loop_length = _count_edges(edges)
        curve_ratio = _calculate_curve_ratio(edges, size)
        logger.info("ループ生成完了: %.3f 秒", time.perf_counter() - step_time)

        if loop_length < 2 * (rows + cols):
            logger.warning("ループ長が不足したため再試行します")
            continue

        step_time = time.perf_counter()
        clues_all = calculate_clues(edges, size)
        logger.info("ヒント計算完了: %.3f 秒", time.perf_counter() - step_time)

        min_hint = max(1, int(rows * cols * MIN_HINT_RATIO.get(difficulty, 0.1)))
        clues = _reduce_clues(clues_all, size, min_hint=min_hint)

        # 0 が縦横に並んでいないか確認
        if _has_zero_adjacent(
            [[v if v is not None else -1 for v in row] for row in clues]
        ):
            logger.warning("0 が隣接したため再試行します")
            continue

        solutions, solver_stats = cast(
            tuple[int, Dict[str, int]],
            count_solutions(
                clues,
                size,
                limit=2,
                return_stats=True,
                step_limit=MAX_SOLVER_STEPS,
            ),
        )
        if solutions != 1:
            logger.warning("解が一意でないためヒントを再計算します")
            clues = cast(List[List[int | None]], [row[:] for row in clues_all])
            solutions, solver_stats = cast(
                tuple[int, Dict[str, int]],
                count_solutions(
                    clues,
                    size,
                    limit=2,
                    return_stats=True,
                    step_limit=MAX_SOLVER_STEPS,
                ),
            )
            if solutions != 1:
                logger.warning("再試行します")
                last_edges = edges
                continue

        puzzle = _build_puzzle_dict(
            size=size,
            edges=edges,
            clues=clues,
            clues_full=clues_all,
            loop_length=loop_length,
            curve_ratio=curve_ratio,
            difficulty=difficulty,
            solver_stats=solver_stats,
            symmetry=symmetry,
            generation_params=generation_params,
            seed_hash=seed_hash,
        )

        # 生成した結果が仕様を満たすか簡易チェック
        if symmetry != "rotational":
            validate_puzzle(puzzle)

        stats = {
            "loop_length": loop_length,
            "hint_count": sum(1 for row in clues for v in row if v is not None),
            "solver_steps": solver_stats["steps"],
            "solver_max_depth": solver_stats["max_depth"],
        }
        logger.info("盤面生成成功: %.3f 秒", time.perf_counter() - start_time)
        if return_stats:
            return puzzle, stats
        return puzzle

    # すべて失敗した場合は最後に計算した edges を使用してフルヒントで返す
    if last_edges is not None:
        clues_all = calculate_clues(last_edges, size)
        curve_ratio_fb = _calculate_curve_ratio(last_edges, size)
        # フォールバックでも 0 の隣接を許さない
        if _has_zero_adjacent(clues_all):
            raise ValueError("0 が隣接しています")
        _, solver_stats = cast(
            tuple[int, Dict[str, int]],
            count_solutions(
                cast(List[List[int | None]], [row[:] for row in clues_all]),
                size,
                limit=2,
                return_stats=True,
                step_limit=MAX_SOLVER_STEPS,
            ),
        )
        puzzle = _build_puzzle_dict(
            size=size,
            edges=last_edges,
            clues=cast(List[List[int | None]], clues_all),
            clues_full=clues_all,
            loop_length=_count_edges(last_edges),
            curve_ratio=curve_ratio_fb,
            difficulty=difficulty,
            solver_stats=solver_stats,
            symmetry=symmetry,
            generation_params=generation_params,
            seed_hash=seed_hash,
        )
        if symmetry != "rotational":
            validate_puzzle(puzzle)
        stats = {
            "loop_length": _count_edges(last_edges),
            "hint_count": sum(1 for row in clues_all for v in row if v is not None),
            "solver_steps": solver_stats["steps"],
            "solver_max_depth": solver_stats["max_depth"],
        }
        logger.info(
            "盤面生成成功(フォールバック): %.3f 秒", time.perf_counter() - start_time
        )
        if return_stats:
            return puzzle, stats
        return puzzle

    raise ValueError("盤面生成に失敗しました")


def generate_puzzle_parallel(
    rows: int,
    cols: int,
    *,
    difficulty: str = "normal",
    seed: int | None = None,
    symmetry: Optional[str] = None,
    return_stats: bool = False,
    jobs: int | None = None,
) -> Puzzle | tuple[Puzzle, Dict[str, int]]:
    """複数プロセスで generate_puzzle を試行して最初の結果を返す"""

    # 初期シードに ``seed`` を使い、プロセス番号でシードをずらして実行する
    # ``jobs`` が ``None`` の場合は CPU コア数を利用する

    if jobs is None:
        jobs = os.cpu_count() or 1

    base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                generate_puzzle,
                rows,
                cols,
                difficulty=difficulty,
                seed=base_seed + i,
                symmetry=symmetry,
                return_stats=return_stats,
            )
            for i in range(jobs)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                for f in futures:
                    f.cancel()
                return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("並列生成失敗: %s", exc)

    raise ValueError("並列生成に失敗しました")


def generate_multiple_puzzles(
    rows: int, cols: int, count_each: int, *, seed: int | None = None
) -> List[Puzzle]:
    """各難易度を同数生成して一覧で返す

    :param rows: 盤面の行数
    :param cols: 盤面の列数
    :param count_each: 各難易度の生成数
    :param seed: 乱数シード。再現したいときに指定する
    """

    if count_each <= 0:
        raise ValueError("count_each は 1 以上を指定してください")

    logger.info(
        "複数盤面生成開始 rows=%d cols=%d count_each=%d",
        rows,
        cols,
        count_each,
    )
    start_time = time.perf_counter()

    puzzles: List[Puzzle] = []
    seed_offset = 0
    # 難易度の順序を固定するためリスト化
    for difficulty in sorted(ALLOWED_DIFFICULTIES):
        for _ in range(count_each):
            puzzle_seed = None if seed is None else seed + seed_offset
            puzzle_obj = generate_puzzle(
                rows, cols, difficulty=difficulty, seed=puzzle_seed
            )
            puzzles.append(cast(Puzzle, puzzle_obj))
            seed_offset += 1

    logger.info("複数盤面生成終了: %.3f 秒", time.perf_counter() - start_time)
    return puzzles


def puzzle_to_ascii(puzzle: Puzzle) -> str:
    """パズル情報から ASCII 形式の盤面を生成して文字列で返す"""

    size_dict = puzzle["size"]
    size = PuzzleSize(rows=size_dict["rows"], cols=size_dict["cols"])
    clues: List[List[int]] = puzzle["clues"]
    edges = puzzle["solutionEdges"]
    horizontal: List[List[bool]] = edges["horizontal"]
    vertical: List[List[bool]] = edges["vertical"]

    lines: List[str] = []
    for r in range(size.rows * 2 + 1):
        if r % 2 == 0:
            # 頂点行。水平線を描画する
            row_idx = r // 2
            line = ""
            for c in range(size.cols):
                line += "+"
                line += "---" if horizontal[row_idx][c] else "   "
            line += "+"
            lines.append(line)
        else:
            # 数字行。縦線とヒント数字を描画する
            row_idx = r // 2
            line = ""
            for c in range(size.cols + 1):
                line += "|" if vertical[row_idx][c] else " "
                if c < size.cols:
                    value = clues[row_idx][c]
                    # None のまま表示すると幅が広がるので 'N' で代用
                    cell = "N" if value is None else str(value)
                    line += f" {cell} "
            lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    # コマンドライン引数を受け取る
    parser = argparse.ArgumentParser(description="スリザーリンク盤面を生成します")
    parser.add_argument("rows", type=int, help="盤面の行数")
    parser.add_argument("cols", type=int, help="盤面の列数")
    parser.add_argument(
        "--difficulty",
        choices=sorted(ALLOWED_DIFFICULTIES),
        default="easy",
        help="難易度ラベル",
    )
    parser.add_argument(
        "--symmetry",
        choices=["rotational"],
        help="回転対称にしたい場合に指定",
    )
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="並列生成プロセス数 (1 なら通常実行)",
    )
    args = parser.parse_args()

    if args.parallel > 1:
        pzl_obj = generate_puzzle_parallel(
            args.rows,
            args.cols,
            difficulty=args.difficulty,
            seed=args.seed,
            symmetry=args.symmetry,
            jobs=args.parallel,
        )
    else:
        pzl_obj = generate_puzzle(
            args.rows,
            args.cols,
            difficulty=args.difficulty,
            seed=args.seed,
            symmetry=args.symmetry,
        )
    pzl = cast(Puzzle, pzl_obj)
    path = save_puzzle(pzl)
    print(f"{path} を作成しました")
    print(puzzle_to_ascii(pzl))
