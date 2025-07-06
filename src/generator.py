"""簡易的なスリザーリンク盤面生成モジュール"""

from __future__ import annotations

import logging
import time
import os
import random
import concurrent.futures
import multiprocessing
import hashlib
from typing import Dict, List, Optional, cast, TYPE_CHECKING

if TYPE_CHECKING:
    # 型チェック時は絶対インポートを使用する
    from src.solver import PuzzleSize, calculate_clues, count_solutions
else:
    try:
        # パッケージ実行時は相対インポート
        from .solver import PuzzleSize, calculate_clues, count_solutions
    except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
        # スクリプトとして直接実行されたときは同じディレクトリからインポートする
        from solver import PuzzleSize, calculate_clues, count_solutions

try:
    # loop_builder モジュールから各種関数を読み込む
    from .loop_builder import (
        _create_empty_edges,
        _generate_random_loop,
        combine_patterns,
        _count_edges,
        _calculate_curve_ratio,
        _apply_rotational_symmetry,
        _apply_vertical_symmetry,
        _apply_horizontal_symmetry,
    )
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from loop_builder import (
        _create_empty_edges,
        _generate_random_loop,
        combine_patterns,
        _count_edges,
        _calculate_curve_ratio,
        _apply_rotational_symmetry,
        _apply_vertical_symmetry,
        _apply_horizontal_symmetry,
    )  # type: ignore

try:
    from .puzzle_io import save_puzzle
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from puzzle_io import save_puzzle

try:
    from .validator import validate_puzzle
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from validator import validate_puzzle

try:
    from .constants import MAX_SOLVER_STEPS
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from constants import MAX_SOLVER_STEPS

try:
    from .puzzle_builder import _reduce_clues, _build_puzzle_dict, _optimize_clues
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from puzzle_builder import _reduce_clues, _build_puzzle_dict, _optimize_clues

try:
    # ``types`` モジュールと名前が衝突しないよう ``puzzle_types`` ファイルで定義
    from .puzzle_types import Puzzle
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from puzzle_types import Puzzle

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """ログ出力の設定を行う関数

    Python の ``logging`` モジュールはアプリの動作状況を
    画面やファイルに出力する仕組みです。ここでは ``basicConfig`` を
    使ってフォーマットと出力レベルをまとめて設定します。

    :param level: 表示するログの重要度。``logging.INFO`` などを指定
    """

    # logging.basicConfig でフォーマットやレベルを一括設定する
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


ALLOWED_DIFFICULTIES = {"easy", "normal", "hard", "expert"}


# difficulty ごとの最小ヒント比率。盤面のセル数に掛けて下限ヒント数を求める
MIN_HINT_RATIO = {
    "easy": 0.3,
    "normal": 0.2,
    "hard": 0.15,
    "expert": 0.1,
}

# 生成失敗時に何回まで再試行するか
# 5 回では盤面サイズによっては失敗することがあったため
# 少し余裕を持たせる
RETRY_LIMIT = 20

# MAX_SOLVER_STEPS と _evaluate_difficulty は constants モジュールへ移動した


def _vertex_degree(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize, r: int, c: int
) -> int:
    """頂点の接続数を求める簡易ヘルパー

    グリッド上の一点(r, c)に接続している線の数を数えます。2 ならループ、
    0 なら線がない状態です。初心者向けに説明すると、これはその点から
    何本の線が出ているかを数える処理です。"""

    deg = 0
    if c < size.cols and edges["horizontal"][r][c]:
        deg += 1
    if c > 0 and edges["horizontal"][r][c - 1]:
        deg += 1
    if r < size.rows and edges["vertical"][r][c]:
        deg += 1
    if r > 0 and edges["vertical"][r - 1][c]:
        deg += 1
    return deg


def _validate_edges(edges: Dict[str, List[List[bool]]], size: PuzzleSize) -> bool:
    """辺情報が単一ループを構成するか確認"""

    for r in range(size.rows + 1):
        for c in range(size.cols + 1):
            # 各頂点の接続本数を調べる
            deg = _vertex_degree(edges, size, r, c)
            # 0 または 2 以外なら分岐しているので NG
            if deg not in (0, 2):
                return False
    return True


def _create_loop(
    size: PuzzleSize,
    rng: random.Random,
    *,
    symmetry: Optional[str],
    theme: Optional[str],
) -> tuple[Dict[str, List[List[bool]]], int, float]:
    """ループを生成し長さと曲率を返す

    ``theme`` が ``"border"`` の場合は外周のみを使った単純なループ、
    ``"pattern"`` の場合は小さなループパターンを敷き詰め、
    ``"maze"`` の場合はランダム生成から曲がりの多いものを選び、
    ``"spiral"`` の場合は曲率比率が最も高い候補を採用する。
    指定がない場合は完全ランダム生成となる。
    """

    # 対称性のあるループは失敗しやすいため複数回試行する
    for _ in range(5):
        edges = _create_empty_edges(size)
        if theme == "border":
            for c in range(size.cols):
                edges["horizontal"][0][c] = True
                edges["horizontal"][size.rows][c] = True
            for r in range(size.rows):
                edges["vertical"][r][0] = True
                edges["vertical"][r][size.cols] = True
        elif theme == "pattern":
            edges = combine_patterns(size, rng)
        elif theme == "maze":
            # ランダムループを複数回生成し、より長く曲がりの多いものを採用する
            best_edges = None
            best_score = -1.0
            for _ in range(10):
                cand = _create_empty_edges(size)
                _generate_random_loop(cand, size, rng)
                if not _validate_edges(cand, size):
                    continue
                length = _count_edges(cand)
                curve = _calculate_curve_ratio(cand, size)
                score = length + curve * 100.0
                if score > best_score:
                    best_edges = cand
                    best_score = score
            if best_edges is not None:
                edges = best_edges
            else:
                _generate_random_loop(edges, size, rng)
        elif theme == "spiral":
            # 曲率比率が最も高いループを採用する
            best_edges = None
            best_curve = -1.0
            for _ in range(10):
                cand = _create_empty_edges(size)
                _generate_random_loop(cand, size, rng)
                if not _validate_edges(cand, size):
                    continue
                curve = _calculate_curve_ratio(cand, size)
                if curve > best_curve:
                    best_edges = cand
                    best_curve = curve
            if best_edges is not None:
                edges = best_edges
            else:
                _generate_random_loop(edges, size, rng)
        else:
            _generate_random_loop(edges, size, rng)

        if symmetry == "rotational":
            _apply_rotational_symmetry(edges, size)
        elif symmetry == "vertical":
            _apply_vertical_symmetry(edges, size)
        elif symmetry == "horizontal":
            _apply_horizontal_symmetry(edges, size)

        if _validate_edges(edges, size):
            loop_length = _count_edges(edges)
            curve_ratio = _calculate_curve_ratio(edges, size)
            return edges, loop_length, curve_ratio

    # すべて失敗した場合は外周だけの単純なループを返す
    edges = _create_empty_edges(size)
    for c in range(size.cols):
        edges["horizontal"][0][c] = True
        edges["horizontal"][size.rows][c] = True
    for r in range(size.rows):
        edges["vertical"][r][0] = True
        edges["vertical"][r][size.cols] = True

    loop_length = _count_edges(edges)
    curve_ratio = _calculate_curve_ratio(edges, size)
    return edges, loop_length, curve_ratio


def _generate_loop_with_symmetry(
    size: PuzzleSize,
    rng: random.Random,
    *,
    symmetry: Optional[str],
    theme: Optional[str],
) -> tuple[Dict[str, List[List[bool]]], int, float]:
    """対称性やテーマを考慮したループを生成する補助関数

    ``_create_loop`` へ処理を委譲し、処理時間の計測とログ出力だけをここで行う。
    """

    start = time.perf_counter()
    edges, loop_length, curve_ratio = _create_loop(
        size, rng, symmetry=symmetry, theme=theme
    )
    logger.info("ループ生成完了: %.3f 秒", time.perf_counter() - start)
    return edges, loop_length, curve_ratio


def _compute_clues_and_optimize(
    edges: Dict[str, List[List[bool]]],
    size: PuzzleSize,
    rng: random.Random,
    *,
    difficulty: str,
    solver_step_limit: int,
    loop_length: int,
    curve_ratio: float,
) -> Optional[tuple[List[List[int | None]], List[List[int]], Dict[str, int]]]:
    """ヒント計算と最適化をまとめて行う補助関数

    解が一意にならない場合は ``None`` を返して再試行を促す。"""

    start = time.perf_counter()
    clues_all = calculate_clues(edges, size)
    logger.info("ヒント計算完了: %.3f 秒", time.perf_counter() - start)

    min_hint = max(1, int(size.rows * size.cols * MIN_HINT_RATIO.get(difficulty, 0.1)))

    clues = _reduce_clues(
        clues_all, size, rng, min_hint=min_hint, step_limit=solver_step_limit
    )

    base_solutions, base_stats = cast(
        tuple[int, Dict[str, int]],
        count_solutions(
            clues,
            size,
            limit=2,
            return_stats=True,
            step_limit=solver_step_limit,
        ),
    )
    if base_solutions != 1:
        logger.warning("解が一意でないため再試行します")
        return None

    clues = _optimize_clues(
        clues,
        clues_all,
        size,
        rng,
        min_hint=min_hint,
        loop_length=loop_length,
        curve_ratio=curve_ratio,
        solver_steps=base_stats["steps"],
        iterations=5,
        step_limit=min(solver_step_limit, 2000),
    )

    solutions, solver_stats = cast(
        tuple[int, Dict[str, int]],
        count_solutions(
            clues,
            size,
            limit=2,
            return_stats=True,
            step_limit=solver_step_limit,
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
                step_limit=solver_step_limit,
            ),
        )
        if solutions != 1:
            logger.warning("再試行します")
            return None

    return clues, clues_all, solver_stats


def generate_puzzle(
    rows: int,
    cols: int,
    difficulty: str = "normal",
    *,
    seed: int | None = None,
    symmetry: Optional[str] = None,
    theme: Optional[str] = None,
    solver_step_limit: int = MAX_SOLVER_STEPS,
    timeout_s: float | None = None,
    return_stats: bool = False,
) -> Puzzle | tuple[Puzzle, Dict[str, int]]:
    """簡易な盤面を生成して返す

    :param rows: 盤面の行数
    :param cols: 盤面の列数
    :param difficulty: 難易度ラベル
    :param seed: 乱数シード。再現したいときに指定する
    :param symmetry: 対称性を指定。"rotational" / "vertical" / "horizontal" のいずれか
    :param theme: 盤面のテーマ。"border", "pattern", "maze", "spiral" のいずれか
    :param timeout_s: 生成処理のタイムアウト秒。None なら無制限
    :param solver_step_limit: ソルバー探索の最大ステップ数
    :param return_stats: True なら生成統計も返す
    :return: 生成したパズル。``return_stats`` が True の場合は
        ``(Puzzle, dict)`` のタプルを返す

    生成結果の ``solverStats`` フィールドには、
    ソルバーが使った探索ステップ数 (``steps``)、
    最大探索深さ (``maxDepth``)、
    頂点次数ルールで枝刈りした回数 (``ruleVertex``)、
    ヒント矛盾で枝刈りした回数 (``ruleClue``) を記録します。
    """

    if difficulty not in ALLOWED_DIFFICULTIES:
        raise ValueError(f"difficulty は {ALLOWED_DIFFICULTIES} のいずれかで指定")

    # 乱数生成器を作成。シードを指定すると結果を再現できる
    rng = random.Random(seed)

    generation_params = {
        "rows": rows,
        "cols": cols,
        "difficulty": difficulty,
        "seed": seed,
        "symmetry": symmetry,
        "theme": theme,
        "solverStepLimit": solver_step_limit,
    }

    seed_hash = hashlib.sha256(str(seed).encode("utf-8")).hexdigest()

    start_time = time.perf_counter()
    logger.info("盤面生成開始: %dx%d difficulty=%s", rows, cols, difficulty)

    size = PuzzleSize(rows=rows, cols=cols)

    last_edges: Dict[str, List[List[bool]]] | None = None
    best_puzzle: Puzzle | None = None
    # 対称性を指定すると成功率が下がるため試行回数を増やす
    retry_limit = RETRY_LIMIT * 50 if symmetry else RETRY_LIMIT
    for attempt in range(retry_limit):
        if timeout_s is not None and time.perf_counter() - start_time > timeout_s:
            if best_puzzle is not None:
                best_puzzle["partial"] = True
                return best_puzzle
            raise TimeoutError("generation timed out")
        edges, loop_length, curve_ratio = _generate_loop_with_symmetry(
            size, rng, symmetry=symmetry, theme=theme
        )

        if loop_length < 2 * (rows + cols):
            logger.warning("ループ長が不足したため再試行します")
            continue

        result = _compute_clues_and_optimize(
            edges,
            size,
            rng,
            difficulty=difficulty,
            solver_step_limit=solver_step_limit,
            loop_length=loop_length,
            curve_ratio=curve_ratio,
        )
        if result is None:
            last_edges = edges
            continue
        clues, clues_all, solver_stats = result

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
            theme=theme,
            generation_params=generation_params,
            seed_hash=seed_hash,
            partial=False,
        )

        # 生成した結果が仕様を満たすか簡易チェック
        try:
            validate_puzzle(puzzle)
        except ValueError as exc:
            # 検証に失敗した場合は再試行
            logger.warning("検証失敗: %s", exc)
            last_edges = edges
            continue

        best_puzzle = puzzle
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
        _, solver_stats = cast(
            tuple[int, Dict[str, int]],
            count_solutions(
                cast(List[List[int | None]], [row[:] for row in clues_all]),
                size,
                limit=2,
                return_stats=True,
                step_limit=solver_step_limit,
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
            theme=theme,
            generation_params=generation_params,
            seed_hash=seed_hash,
            partial=True,
        )
        # フォールバック結果でも必ず検証を行う
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

    if best_puzzle is not None:
        best_puzzle["partial"] = True
        return best_puzzle

    raise ValueError("盤面生成に失敗しました")


def _safe_generate_parallel(
    args: tuple,
) -> Puzzle | tuple[Puzzle, Dict[str, int]] | Exception:
    """``generate_puzzle`` をラップし例外を返すヘルパー関数"""

    (
        seed_offset,
        rows,
        cols,
        difficulty,
        base_seed,
        symmetry,
        theme,
        solver_step_limit,
        timeout_s,
        return_stats,
    ) = args

    try:
        return generate_puzzle(
            rows,
            cols,
            difficulty=difficulty,
            seed=base_seed + seed_offset,
            symmetry=symmetry,
            theme=theme,
            timeout_s=timeout_s,
            solver_step_limit=solver_step_limit,
            return_stats=return_stats,
        )
    except Exception as exc:  # noqa: BLE001
        return exc


def generate_puzzle_parallel(
    rows: int,
    cols: int,
    *,
    difficulty: str = "normal",
    seed: int | None = None,
    symmetry: Optional[str] = None,
    theme: Optional[str] = None,
    solver_step_limit: int = MAX_SOLVER_STEPS,
    timeout_s: float | None = None,
    return_stats: bool = False,
    jobs: int | None = None,
    worker_log_level: int = logging.WARNING,
) -> Puzzle | tuple[Puzzle, Dict[str, int]]:
    """複数プロセスで ``generate_puzzle`` を試行し最初に成功した盤面を返す

    並列生成の目的は時間短縮のため、1 つの盤面ができた時点で他の
    ワーカーはキャンセルして処理を終える。

    :param theme: 盤面のテーマを指定する文字列
    :param solver_step_limit: ソルバー探索の最大ステップ数
    :param timeout_s: 生成処理のタイムアウト秒数
    :param worker_log_level: 並列処理のログレベル。デフォルトでは WARNING
        以上のみ表示する
    """

    # 初期シードに ``seed`` を使い、プロセス番号でシードをずらして実行する
    # ``jobs`` が ``None`` の場合は CPU コア数を利用する

    if jobs is None:
        jobs = os.cpu_count() or 1

    base_seed = seed if seed is not None else random.randint(0, 2**32 - 1)

    # multiprocessing.Pool を使い、最初に成功した結果を得たら残りの
    # ワーカーを即座に終了させる
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=jobs, initializer=setup_logging, initargs=(worker_log_level,)
    ) as pool:
        iterator = pool.imap_unordered(
            _safe_generate_parallel,
            (
                (
                    i,
                    rows,
                    cols,
                    difficulty,
                    base_seed,
                    symmetry,
                    theme,
                    solver_step_limit,
                    timeout_s,
                    return_stats,
                )
                for i in range(jobs)
            ),
        )
        result = None
        failures = 0
        for res in iterator:
            if isinstance(res, Exception):
                logger.warning("並列生成失敗: %s", res)
                failures += 1
                if failures >= jobs:
                    raise ValueError("並列生成に失敗しました")
                continue
            result = res
            break

        # 1 件取得したら残りのプロセスを終了させる
        pool.terminate()

    if result is None:
        raise ValueError("並列生成に失敗しました")
    return result


def generate_multiple_puzzles(
    rows: int,
    cols: int,
    count_each: int,
    *,
    seed: int | None = None,
    jobs: int | None = None,
    worker_log_level: int = logging.WARNING,
) -> List[Puzzle]:
    """各難易度を同数生成して一覧で返す

    :param rows: 盤面の行数
    :param cols: 盤面の列数
    :param count_each: 各難易度の生成数
    :param seed: 乱数シード。再現したいときに指定する
    :param worker_log_level: 並列処理のログレベル。WARNING 以上のみ表示する
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

    if jobs is None:
        jobs = 1

    # jobs が 1 の場合は従来どおり逐次生成
    if jobs <= 1:
        # 難易度の順序を固定するためリスト化
        for difficulty in sorted(ALLOWED_DIFFICULTIES):
            generated = 0
            while generated < count_each:
                puzzle_seed = None if seed is None else seed + seed_offset
                try:
                    puzzle_obj = generate_puzzle(
                        rows, cols, difficulty=difficulty, seed=puzzle_seed
                    )
                except ValueError as exc:
                    # まれに 0 が隣接して生成に失敗することがあるため再試行
                    logger.warning("%s の生成失敗: %s", difficulty, exc)
                    seed_offset += 1
                    continue
                puzzles.append(cast(Puzzle, puzzle_obj))
                generated += 1
                seed_offset += 1
    else:
        # 並列に各パズル生成を実行する
        if seed is None:
            base_seed = random.randint(0, 2**32 - 1)
        else:
            base_seed = seed
        futures = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs,
            initializer=setup_logging,
            initargs=(worker_log_level,),
        ) as executor:
            for difficulty in sorted(ALLOWED_DIFFICULTIES):
                for _ in range(count_each):
                    puzzle_seed = base_seed + seed_offset
                    seed_offset += 1
                    futures.append(
                        executor.submit(
                            generate_puzzle,
                            rows,
                            cols,
                            difficulty=difficulty,
                            seed=puzzle_seed,
                        )
                    )
            for future in concurrent.futures.as_completed(futures):
                try:
                    puzzle_obj = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("並列生成失敗: %s", exc)
                    continue
                puzzles.append(cast(Puzzle, puzzle_obj))

        expected = count_each * len(ALLOWED_DIFFICULTIES)
        if len(puzzles) < expected:
            raise ValueError("並列生成に失敗しました")

    logger.info("複数盤面生成終了: %.3f 秒", time.perf_counter() - start_time)
    return puzzles


def puzzle_to_ascii(puzzle: Puzzle) -> str:
    """パズル情報を簡易的なテキスト盤面へ変換する"""

    # ``+`` や ``|`` を使って盤面を描くシンプルな関数です。

    size_dict = puzzle["size"]
    size = PuzzleSize(rows=size_dict["rows"], cols=size_dict["cols"])
    clues: List[List[int]] = puzzle["clues"]
    edges = puzzle["solutionEdges"]
    horizontal: List[List[bool]] = edges["horizontal"]
    vertical: List[List[bool]] = edges["vertical"]

    lines: List[str] = []
    for r in range(size.rows * 2 + 1):
        if r % 2 == 0:
            # 偶数行は頂点を表す行。水平線を描画する
            row_idx = r // 2
            line = ""
            for c in range(size.cols):
                line += "+"
                line += "---" if horizontal[row_idx][c] else "   "
            line += "+"
            lines.append(line)
        else:
            # 奇数行は数字を配置する行。縦線とヒント数字を描画
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

    # ログ設定を行う。デフォルトは INFO レベル
    setup_logging()

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
        choices=["rotational", "vertical", "horizontal"],
        help="対称性を指定",
    )
    parser.add_argument(
        "--theme",
        choices=["border", "pattern", "maze", "spiral"],
        help="盤面のテーマを指定",
    )
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="タイムアウト秒数 (指定しない場合は無制限)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="並列生成プロセス数 (1 なら通常実行)",
    )
    parser.add_argument(
        "--step-limit",
        type=int,
        default=MAX_SOLVER_STEPS,
        help="ソルバーの最大ステップ数",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="プロファイル結果を profile.prof に保存する",
    )
    args = parser.parse_args()

    func_kwargs = {
        "difficulty": args.difficulty,
        "seed": args.seed,
        "symmetry": args.symmetry,
        "theme": args.theme,
        "solver_step_limit": args.step_limit,
        "timeout_s": args.timeout,
    }

    if args.parallel > 1:
        func = generate_puzzle_parallel
        func_kwargs.update({"jobs": args.parallel, "worker_log_level": logging.WARNING})
    else:
        func = generate_puzzle

    if args.profile:
        # cProfile でプロファイルを取得し profile.prof に書き出す
        import cProfile

        profiler = cProfile.Profile()
        pzl_obj = profiler.runcall(func, args.rows, args.cols, **func_kwargs)
        profiler.dump_stats("profile.prof")
    else:
        pzl_obj = func(args.rows, args.cols, **func_kwargs)
    pzl = cast(Puzzle, pzl_obj)
    path = save_puzzle(pzl)
    print(f"{path} を作成しました")
    print(puzzle_to_ascii(pzl))
