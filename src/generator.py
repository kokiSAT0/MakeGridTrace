"""簡易的なスリザーリンク盤面生成モジュール"""

from __future__ import annotations

import logging
import time
import random
import concurrent.futures
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
    # loop_builder から必要な関数のみ読み込む
    from .loop_builder import _count_edges, _calculate_curve_ratio
    from .loop_wilson import generate_loop
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from loop_builder import _count_edges, _calculate_curve_ratio  # type: ignore
    from loop_wilson import generate_loop

try:
    from .puzzle_io import save_puzzle
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from puzzle_io import save_puzzle

try:
    from .validator import validate_puzzle
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from validator import validate_puzzle


try:
    from .puzzle_builder import _reduce_clues, _build_puzzle_dict, _optimize_clues
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from puzzle_builder import _reduce_clues, _build_puzzle_dict, _optimize_clues

try:
    # ``types`` モジュールと名前が衝突しないよう ``puzzle_types`` ファイルで定義
    from .puzzle_types import Puzzle
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from puzzle_types import Puzzle

try:
    from . import sat_unique
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    import sat_unique

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

# difficulty ごとの最大ヒント比率。盤面セル数に掛けた上限ヒント数を表す
MAX_HINT_RATIO = {
    "easy": 0.38,
    "normal": 0.30,
    "hard": 0.22,
    "expert": 0.18,
}

# 生成失敗時に何回まで再試行するか
# 5 回では盤面サイズによっては失敗することがあったため
# 少し余裕を持たせる
RETRY_LIMIT = 20

# 難易度推定ロジックは constants モジュールへ分離している


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
    edges, loop_length, curve_ratio = generate_loop(
        size, rng, symmetry=symmetry, theme=theme
    )
    logger.info("ループ生成完了: %.3f 秒", time.perf_counter() - start)
    return edges, loop_length, curve_ratio


def _inject_clue_patterns(
    clues: List[List[int | None]], size: PuzzleSize, rng: random.Random
) -> bool:
    """222 や 33 のヒント列を確率的に挿入する簡易処理"""

    injected = False
    prob = 0.07  # 5% から 10% 程度の確率
    # 横方向に 222 を入れる
    if rng.random() < prob and size.cols >= 3:
        r = rng.randrange(size.rows)
        c = rng.randrange(size.cols - 2)
        if all(clues[r][c + i] is None for i in range(3)):
            clues[r][c] = clues[r][c + 1] = clues[r][c + 2] = 2
            injected = True
    # 横方向に 33 を入れる
    if rng.random() < prob and size.cols >= 2:
        r = rng.randrange(size.rows)
        c = rng.randrange(size.cols - 1)
        if clues[r][c] is None and clues[r][c + 1] is None:
            clues[r][c] = clues[r][c + 1] = 3
            injected = True

    # 0 が隣接しないよう調整
    if injected:
        for r in range(size.rows):
            for c in range(size.cols):
                if clues[r][c] == 0:
                    if r + 1 < size.rows and clues[r + 1][c] == 0:
                        clues[r + 1][c] = None
                    if c + 1 < size.cols and clues[r][c + 1] == 0:
                        clues[r][c + 1] = None
    return injected


def _compute_clues_and_optimize(
    edges: Dict[str, List[List[bool]]],
    size: PuzzleSize,
    rng: random.Random,
    *,
    difficulty: str,
    max_hint: int,
    solver_step_limit: int,
    loop_length: int,
    curve_ratio: float,
) -> Optional[tuple[List[List[int | None]], List[List[int]], Dict[str, int]]]:
    """ヒント計算と最適化をまとめて行う補助関数

    解が一意にならない場合は ``None`` を返して再試行を促す。

    :param max_hint: ヒント数の上限
    """

    start = time.perf_counter()
    clues_all = calculate_clues(edges, size)
    logger.info("ヒント計算完了: %.3f 秒", time.perf_counter() - start)

    min_hint = max(1, int(size.rows * size.cols * MIN_HINT_RATIO.get(difficulty, 0.1)))

    clues = _reduce_clues(
        clues_all,
        size,
        rng,
        min_hint=min_hint,
        max_hint=max_hint,
        step_limit=solver_step_limit,
    )

    # PySAT で一意解か確認
    if not sat_unique.is_unique(clues, size):
        logger.warning("解が一意でないため再試行します")
        return None

    # 解析統計は既存ソルバーで取得する
    base_stats = cast(
        Dict[str, int],
        count_solutions(
            clues,
            size,
            limit=2,
            return_stats=True,
            step_limit=solver_step_limit,
        )[1],
    )

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

    before_inject = [row[:] for row in clues]
    if _inject_clue_patterns(clues, size, rng):
        # 注入後に解の一意性を確認する
        if not sat_unique.is_unique(clues, size):
            clues = before_inject

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
    if not sat_unique.is_unique(clues, size):
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
        if not sat_unique.is_unique(clues, size):
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
    solver_step_limit: int | None = None,
    timeout_s: float | None = None,
    return_stats: bool = False,
) -> Puzzle | tuple[Puzzle, Dict[str, int]]:
    """簡易な盤面を生成して返す

    :param rows: 盤面の行数
    :param cols: 盤面の列数
    :param difficulty: 難易度ラベル
    :param seed: 乱数シード。再現したいときに指定する
    :param symmetry: 対称性を指定。"rotational" / "vertical" / "horizontal" のいずれか
    :param theme: 盤面のテーマ。"border", "pattern", "maze", "spiral", "figure8",
        "labyrinth" のいずれか
    :param timeout_s: 生成処理のタイムアウト秒。None なら無制限
    :param solver_step_limit: ソルバー探索の最大ステップ数。``None`` の場合は
        ``rows * cols * 25`` を利用する
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

    if solver_step_limit is None:
        # 盤面サイズに応じて上限を決める
        solver_step_limit = rows * cols * 25

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

    # ヒント数の上限を難易度に応じて計算
    max_hint = int(size.rows * size.cols * MAX_HINT_RATIO.get(difficulty, 1.0))

    last_edges: Dict[str, List[List[bool]]] | None = None
    best_puzzle: Puzzle | None = None
    # 対称性を指定すると成功率が下がるため試行回数を増やす
    retry_limit = RETRY_LIMIT * 50 if symmetry else RETRY_LIMIT
    for attempt in range(retry_limit):
        if timeout_s is not None and time.perf_counter() - start_time > timeout_s:
            if best_puzzle is not None:
                best_puzzle["partial"] = True
                best_puzzle["reason"] = "timeout"
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
            max_hint=max_hint,
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


def generate_puzzle_parallel(
    rows: int,
    cols: int,
    *,
    difficulty: str = "normal",
    seed: int | None = None,
    symmetry: Optional[str] = None,
    theme: Optional[str] = None,
    solver_step_limit: int | None = None,
    timeout_s: float | None = None,
    return_stats: bool = False,
    jobs: int | None = None,
    worker_log_level: int = logging.WARNING,
) -> Puzzle | tuple[Puzzle, Dict[str, int]]:
    """``generator_parallel`` モジュール経由で並列生成を実行する"""

    # 実際の並列処理は ``generator_parallel`` に委譲する
    from . import generator_parallel as gp

    kwargs = {
        "difficulty": difficulty,
        "seed": seed,
        "symmetry": symmetry,
        "theme": theme,
        "solver_step_limit": solver_step_limit,
        "timeout_s": timeout_s,
        "return_stats": return_stats,
        "jobs": jobs,
        "worker_log_level": worker_log_level,
    }

    return gp.generate_puzzle_parallel(rows, cols, **kwargs)


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

        counts = {d: 0 for d in ALLOWED_DIFFICULTIES}
        retry_round = 0
        while any(c < count_each for c in counts.values()):
            if retry_round > RETRY_LIMIT:
                raise ValueError("並列生成に失敗しました")
            retry_round += 1

            futures: dict[concurrent.futures.Future[Puzzle], str] = {}
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=jobs,
                initializer=setup_logging,
                initargs=(worker_log_level,),
            ) as executor:
                for diff in sorted(ALLOWED_DIFFICULTIES):
                    remain = count_each - counts[diff]
                    for _ in range(remain):
                        puzzle_seed = base_seed + seed_offset
                        seed_offset += 1
                        future = executor.submit(
                            generate_puzzle,
                            rows,
                            cols,
                            difficulty=diff,
                            seed=puzzle_seed,
                        )
                        futures[future] = diff

                for future in concurrent.futures.as_completed(futures):
                    diff = futures[future]
                    try:
                        puzzle_obj = future.result()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("並列生成失敗: %s", exc)
                        continue
                    puzzles.append(cast(Puzzle, puzzle_obj))
                    counts[diff] += 1

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
        choices=["border", "pattern", "maze", "spiral", "figure8", "labyrinth"],
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
        default=None,
        help="ソルバーの最大ステップ数 (未指定なら自動計算)",
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
