"""簡易的なスリザーリンク盤面生成モジュール"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import time
from pathlib import Path
import json
import random
from typing import Any, Dict, List


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class PuzzleSize:
    """盤面サイズを表すデータクラス"""

    rows: int
    cols: int


Puzzle = Dict[str, Any]

# JSON スキーマのバージョン
SCHEMA_VERSION = "2.0"


def _create_empty_edges(size: PuzzleSize) -> Dict[str, List[List[bool]]]:
    """solutionEdges フィールド用の空の二次元配列を作成する"""
    horizontal = [[False for _ in range(size.cols)] for _ in range(size.rows + 1)]
    vertical = [[False for _ in range(size.cols + 1)] for _ in range(size.rows)]
    return {"horizontal": horizontal, "vertical": vertical}


def _generate_random_loop(edges: Dict[str, List[List[bool]]], size: PuzzleSize) -> None:
    """外周だけではないランダムなループを生成する"""

    # 頂点ごとの接続数を管理する配列
    degrees = [[0 for _ in range(size.cols + 1)] for _ in range(size.rows + 1)]

    def edge_exists(a: tuple[int, int], b: tuple[int, int]) -> bool:
        """2点間の辺がすでに存在するか判定"""
        if a[0] == b[0]:
            r = a[0]
            c = min(a[1], b[1])
            return edges["horizontal"][r][c]
        else:
            c = a[1]
            r = min(a[0], b[0])
            return edges["vertical"][r][c]

    def add_edge(a: tuple[int, int], b: tuple[int, int]) -> None:
        """2点を結ぶ辺を追加する"""
        if a[0] == b[0]:
            r = a[0]
            c = min(a[1], b[1])
            edges["horizontal"][r][c] = True
        else:
            c = a[1]
            r = min(a[0], b[0])
            edges["vertical"][r][c] = True
        degrees[a[0]][a[1]] += 1
        degrees[b[0]][b[1]] += 1

    def remove_edge(a: tuple[int, int], b: tuple[int, int]) -> None:
        """2点を結ぶ辺を削除する"""
        if a[0] == b[0]:
            r = a[0]
            c = min(a[1], b[1])
            edges["horizontal"][r][c] = False
        else:
            c = a[1]
            r = min(a[0], b[0])
            edges["vertical"][r][c] = False
        degrees[a[0]][a[1]] -= 1
        degrees[b[0]][b[1]] -= 1

    # 目標ループ長はハード制約 H-7 に従い盤面周長の 2 倍以上とする
    # 盤面周長とは rows + cols を 2 倍した数で、外周を一周する長さを指す
    min_len = max(2 * (size.rows + size.cols), 4)

    # バックトラックでランダムなループを構築する
    start = (random.randint(0, size.rows), random.randint(0, size.cols))
    path: list[tuple[int, int]] = [start]

    def dfs(current: tuple[int, int]) -> bool:
        if len(path) >= min_len and current == start and len(path) > 1:
            return True

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr <= size.rows and 0 <= nc <= size.cols):
                continue
            nxt = (nr, nc)
            if degrees[nr][nc] >= 2 or degrees[current[0]][current[1]] >= 2:
                continue
            if edge_exists(current, nxt):
                continue
            # スタート地点に戻る場合は最小長さを満たすか確認
            if nxt == start and len(path) + 1 < min_len:
                continue

            add_edge(current, nxt)
            path.append(nxt)
            if dfs(nxt):
                return True
            # 戻る処理
            path.pop()
            remove_edge(current, nxt)
        return False

    if not dfs(start):
        # 失敗したら外周ループを生成してお茶を濁す
        for c in range(size.cols):
            edges["horizontal"][0][c] = True
            edges["horizontal"][size.rows][c] = True
        for r in range(size.rows):
            edges["vertical"][r][0] = True
            edges["vertical"][r][size.cols] = True


def _calculate_clues(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> List[List[int]]:
    """solutionEdges からヒント数字を計算する"""
    clues = [[0 for _ in range(size.cols)] for _ in range(size.rows)]
    for r in range(size.rows):
        for c in range(size.cols):
            count = 0
            if edges["horizontal"][r][c]:
                count += 1
            if edges["horizontal"][r + 1][c]:
                count += 1
            if edges["vertical"][r][c]:
                count += 1
            if edges["vertical"][r][c + 1]:
                count += 1
            clues[r][c] = count
    return clues


def _count_edges(edges: Dict[str, List[List[bool]]]) -> int:
    """ループに含まれる辺の総数を数える"""

    # True の数をすべて合計することで長さを求める
    return sum(sum(row) for row in edges["horizontal"]) + sum(
        sum(row) for row in edges["vertical"]
    )


def _calculate_curve_ratio(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> float:
    """ループ中の曲がり角割合を計算する関数"""

    curve_count = 0
    for r in range(size.rows + 1):
        for c in range(size.cols + 1):
            connections = []
            if c < size.cols and edges["horizontal"][r][c]:
                connections.append("h")
            if c > 0 and edges["horizontal"][r][c - 1]:
                connections.append("h")
            if r < size.rows and edges["vertical"][r][c]:
                connections.append("v")
            if r > 0 and edges["vertical"][r - 1][c]:
                connections.append("v")
            if (
                len(connections) == 2
                and connections.count("h") == 1
                and connections.count("v") == 1
            ):
                curve_count += 1
    total = _count_edges(edges)
    return curve_count / total if total > 0 else 0.0


ALLOWED_DIFFICULTIES = {"easy", "normal", "hard", "expert"}


# difficulty ごとの最小ヒント比率。盤面のセル数に掛けて下限ヒント数を求める
MIN_HINT_RATIO = {
    "easy": 0.3,
    "normal": 0.2,
    "hard": 0.15,
    "expert": 0.1,
}

# 生成失敗時に何回まで再試行するか
RETRY_LIMIT = 3

# ソルバーが探索する最大ステップ数。超えると途中で打ち切る
MAX_SOLVER_STEPS = 100000


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


def _count_solutions(
    clues: List[List[int | None]],
    size: PuzzleSize,
    *,
    limit: int = 2,
    return_stats: bool = False,
    step_limit: int | None = None,
) -> int | tuple[int, Dict[str, int]]:
    """バックトラックで解の個数を数える簡易ソルバー

    :param return_stats: True のとき解析ステップ数などを返す
    :param step_limit: ステップ数の上限。None なら制限なし
    """

    if step_limit is None and return_stats:
        step_limit = MAX_SOLVER_STEPS

    # 辺リストとセル・頂点の参照テーブルを構築
    edges_list: list[tuple[str, int, int]] = []
    cell_edges: list[list[list[int]]] = [
        [[] for _ in range(size.cols)] for _ in range(size.rows)
    ]
    vertex_edges: list[list[list[int]]] = [
        [[] for _ in range(size.cols + 1)] for _ in range(size.rows + 1)
    ]

    def add_edge(t: str, r: int, c: int) -> None:
        idx = len(edges_list)
        edges_list.append((t, r, c))
        if t == "h":
            v1 = (r, c)
            v2 = (r, c + 1)
            if r < size.rows:
                cell_edges[r][c].append(idx)
            if r > 0:
                cell_edges[r - 1][c].append(idx)
        else:
            v1 = (r, c)
            v2 = (r + 1, c)
            if c < size.cols:
                cell_edges[r][c].append(idx)
            if c > 0:
                cell_edges[r][c - 1].append(idx)
        vertex_edges[v1[0]][v1[1]].append(idx)
        vertex_edges[v2[0]][v2[1]].append(idx)

    for r in range(size.rows + 1):
        for c in range(size.cols):
            add_edge("h", r, c)
    for r in range(size.rows):
        for c in range(size.cols + 1):
            add_edge("v", r, c)

    n = len(edges_list)
    edge_state = [False] * n
    vertex_degree = [[0 for _ in range(size.cols + 1)] for _ in range(size.rows + 1)]
    cell_count = [[0 for _ in range(size.cols)] for _ in range(size.rows)]
    cell_unknown = [[4 for _ in range(size.cols)] for _ in range(size.rows)]

    solutions = 0
    steps = 0
    max_depth = 0

    def dfs(idx: int, depth: int) -> None:
        nonlocal solutions, steps, max_depth
        steps += 1
        if depth > max_depth:
            max_depth = depth
        if step_limit is not None and steps > step_limit:
            return
        if solutions >= limit:
            return
        if idx == n:
            # すべて決めたので条件を確認
            for r in range(size.rows + 1):
                for c in range(size.cols + 1):
                    if vertex_degree[r][c] not in (0, 2):
                        return
            for r in range(size.rows):
                for c in range(size.cols):
                    clue = clues[r][c]
                    if clue is not None and cell_count[r][c] != clue:
                        return

            horizontal = [
                [False for _ in range(size.cols)] for _ in range(size.rows + 1)
            ]
            vertical = [[False for _ in range(size.cols + 1)] for _ in range(size.rows)]
            for i, (t, r, c) in enumerate(edges_list):
                if not edge_state[i]:
                    continue
                if t == "h":
                    horizontal[r][c] = True
                else:
                    vertical[r][c] = True
            puzzle = {
                "size": {"rows": size.rows, "cols": size.cols},
                "solutionEdges": {"horizontal": horizontal, "vertical": vertical},
                "clues": clues,
            }
            try:
                validate_puzzle(puzzle)
            except ValueError:
                return
            solutions += 1
            return

        t, r, c = edges_list[idx]
        for val in (False, True):
            edge_state[idx] = val
            vertices = [(r, c), (r, c + 1)] if t == "h" else [(r, c), (r + 1, c)]
            cells = []
            if t == "h":
                if r < size.rows:
                    cells.append((r, c))
                if r > 0:
                    cells.append((r - 1, c))
            else:
                if c < size.cols:
                    cells.append((r, c))
                if c > 0:
                    cells.append((r, c - 1))

            for vr, vc in vertices:
                vertex_degree[vr][vc] += int(val)

            for cr, cc in cells:
                cell_unknown[cr][cc] -= 1
                if val:
                    cell_count[cr][cc] += 1

            # 部分的な矛盾がないか確認
            ok = True
            for vr, vc in vertices:
                if vertex_degree[vr][vc] > 2:
                    ok = False
                    break
            if ok:
                for cr, cc in cells:
                    clue = clues[cr][cc]
                    if clue is None:
                        continue
                    used = cell_count[cr][cc]
                    unknown = cell_unknown[cr][cc]
                    if used > clue or used + unknown < clue:
                        ok = False
                        break
            if ok:
                dfs(idx + 1, depth + 1)

            for vr, vc in vertices:
                vertex_degree[vr][vc] -= int(val)
            for cr, cc in cells:
                if val:
                    cell_count[cr][cc] -= 1
                cell_unknown[cr][cc] += 1
        edge_state[idx] = False

    dfs(0, 0)
    if return_stats:
        return solutions, {"steps": steps, "max_depth": max_depth}
    return solutions


def _reduce_clues(
    clues: List[List[int]], size: PuzzleSize, *, min_hint: int
) -> List[List[int | None]]:
    """ヒントをランダムに削減して一意性を保つ"""

    result = [[v for v in row] for row in clues]
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
            or _count_solutions(result, size, limit=2, step_limit=10000) != 1
        ):
            result[r][c] = original

    return result


def generate_puzzle(
    rows: int,
    cols: int,
    difficulty: str = "normal",
    *,
    seed: int | None = None,
    return_stats: bool = False,
) -> Puzzle | tuple[Puzzle, Dict[str, int]]:
    """簡易な盤面を生成して返す

    :param rows: 盤面の行数
    :param cols: 盤面の列数
    :param difficulty: 難易度ラベル
    :param seed: 乱数シード。再現したいときに指定する
    :param return_stats: True なら生成統計も返す
    """

    if difficulty not in ALLOWED_DIFFICULTIES:
        raise ValueError(f"difficulty は {ALLOWED_DIFFICULTIES} のいずれかで指定")

    if seed is not None:
        random.seed(seed)

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
        loop_length = _count_edges(edges)
        curve_ratio = _calculate_curve_ratio(edges, size)
        logger.info("ループ生成完了: %.3f 秒", time.perf_counter() - step_time)

        if loop_length < 2 * (rows + cols):
            logger.warning("ループ長が不足したため再試行します")
            continue

        step_time = time.perf_counter()
        clues_all = _calculate_clues(edges, size)
        logger.info("ヒント計算完了: %.3f 秒", time.perf_counter() - step_time)

        min_hint = max(1, int(rows * cols * MIN_HINT_RATIO.get(difficulty, 0.1)))
        clues = _reduce_clues(clues_all, size, min_hint=min_hint)

        def zero_adjacent(cl: List[List[int | None]]) -> bool:
            for rr in range(size.rows):
                for cc in range(size.cols):
                    if cl[rr][cc] == 0:
                        if rr + 1 < size.rows and cl[rr + 1][cc] == 0:
                            return True
                        if cc + 1 < size.cols and cl[rr][cc + 1] == 0:
                            return True
            return False

        if zero_adjacent(clues):
            logger.warning("0 が隣接したため再試行します")
            continue

        sol_result = _count_solutions(
            clues, size, limit=2, return_stats=True, step_limit=MAX_SOLVER_STEPS
        )
        solutions, solver_stats = sol_result
        if solutions != 1:
            logger.warning("解が一意でないためヒントを再計算します")
            clues = clues_all
            sol_result = _count_solutions(
                clues, size, limit=2, return_stats=True, step_limit=MAX_SOLVER_STEPS
            )
            solutions, solver_stats = sol_result
            if solutions != 1:
                logger.warning("再試行します")
                last_edges = edges
                continue

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        puzzle: Puzzle = {
            "schemaVersion": SCHEMA_VERSION,
            "id": f"sl_{rows}x{cols}_{difficulty}_{timestamp}",
            "size": {"rows": rows, "cols": cols},
            "clues": clues,
            "solutionEdges": edges,
            "loopStats": {"length": loop_length, "curveRatio": curve_ratio},
            "difficulty": difficulty,
            "difficultyEval": _evaluate_difficulty(
                solver_stats["steps"], solver_stats["max_depth"]
            ),
            "createdBy": "auto-gen-v1",
            "createdAt": datetime.utcnow().date().isoformat(),
        }

        # 生成した結果が仕様を満たすか簡易チェック
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
        clues_all = _calculate_clues(last_edges, size)
        curve_ratio_fb = _calculate_curve_ratio(last_edges, size)
        # フォールバックでも 0 の隣接を許さない
        for rr in range(size.rows):
            for cc in range(size.cols):
                if clues_all[rr][cc] == 0:
                    if rr + 1 < size.rows and clues_all[rr + 1][cc] == 0:
                        raise ValueError("0 が縦に隣接しています")
                    if cc + 1 < size.cols and clues_all[rr][cc + 1] == 0:
                        raise ValueError("0 が横に隣接しています")
        sol_result = _count_solutions(
            clues_all, size, limit=2, return_stats=True, step_limit=MAX_SOLVER_STEPS
        )
        _, solver_stats = sol_result
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        puzzle = {
            "schemaVersion": SCHEMA_VERSION,
            "id": f"sl_{rows}x{cols}_{difficulty}_{timestamp}",
            "size": {"rows": rows, "cols": cols},
            "clues": clues_all,
            "solutionEdges": last_edges,
            "loopStats": {
                "length": _count_edges(last_edges),
                "curveRatio": curve_ratio_fb,
            },
            "difficulty": difficulty,
            "difficultyEval": _evaluate_difficulty(
                solver_stats["steps"], solver_stats["max_depth"]
            ),
            "createdBy": "auto-gen-v1",
            "createdAt": datetime.utcnow().date().isoformat(),
        }
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


def save_puzzle(puzzle: Puzzle, directory: str | Path = "data") -> Path:
    """パズルを JSON 形式で保存する

    :param puzzle: generate_puzzle の戻り値
    :param directory: 保存先ディレクトリ
    :return: 保存したファイルのパス
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    # 出力ファイル名は常に map_gridtrace.json とする
    # 複数生成する場合はディレクトリを変えるなどで調整する想定
    file_path = path / "map_gridtrace.json"
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(puzzle, fp, ensure_ascii=False, indent=2)
    logger.info("パズルを保存しました: %s", file_path)
    return file_path


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
            puzzles.append(
                generate_puzzle(rows, cols, difficulty=difficulty, seed=puzzle_seed)
            )
            seed_offset += 1

    logger.info("複数盤面生成終了: %.3f 秒", time.perf_counter() - start_time)
    return puzzles


def save_puzzles(puzzles: List[Puzzle], directory: str | Path = "data") -> Path:
    """複数のパズルをひとつの JSON ファイルに保存する"""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "map_gridtrace.json"
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(puzzles, fp, ensure_ascii=False, indent=2)
    logger.info("複数パズルを保存しました: %s", file_path)
    return file_path


def validate_puzzle(puzzle: Puzzle) -> None:
    """パズルデータの整合性を簡易チェックする関数

    H-7 ループ長、H-8 0 の隣接禁止、H-9 曲率比率を含む
    """

    # size フィールドの検証
    size_dict = puzzle.get("size")
    if not isinstance(size_dict, dict):
        raise ValueError("size フィールドが存在しません")
    size = PuzzleSize(rows=size_dict["rows"], cols=size_dict["cols"])

    edges = puzzle.get("solutionEdges")
    if not isinstance(edges, dict):
        raise ValueError("solutionEdges フィールドが存在しません")

    horizontal = edges.get("horizontal")
    vertical = edges.get("vertical")
    if (
        not isinstance(horizontal, list)
        or not isinstance(vertical, list)
        or len(horizontal) != size.rows + 1
        or len(vertical) != size.rows
    ):
        raise ValueError("solutionEdges のサイズが盤面サイズと一致しません")

    for row in horizontal:
        if len(row) != size.cols:
            raise ValueError("horizontal 配列の列数が不正です")
    for row in vertical:
        if len(row) != size.cols + 1:
            raise ValueError("vertical 配列の列数が不正です")

    # ループ条件の確認
    edge_count = 0
    degrees = [[0 for _ in range(size.cols + 1)] for _ in range(size.rows + 1)]

    for r in range(size.rows + 1):
        for c in range(size.cols):
            if horizontal[r][c]:
                edge_count += 1
                degrees[r][c] += 1
                degrees[r][c + 1] += 1
    for r in range(size.rows):
        for c in range(size.cols + 1):
            if vertical[r][c]:
                edge_count += 1
                degrees[r][c] += 1
                degrees[r + 1][c] += 1

    start = None
    for r in range(size.rows + 1):
        for c in range(size.cols + 1):
            d = degrees[r][c]
            if d not in (0, 2):
                raise ValueError("ループが分岐または交差しています")
            if d == 2 and start is None:
                start = (r, c)

    if start is None:
        raise ValueError("ループが存在しません")

    # BFS でループの連結性を確認
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    queue = [start]
    visited_vertices = {start}

    def neighbors(r: int, c: int) -> list[tuple[int, int]]:
        result = []
        if c < size.cols and horizontal[r][c]:
            result.append((r, c + 1))
        if c > 0 and horizontal[r][c - 1]:
            result.append((r, c - 1))
        if r < size.rows and vertical[r][c]:
            result.append((r + 1, c))
        if r > 0 and vertical[r - 1][c]:
            result.append((r - 1, c))
        return result

    while queue:
        r, c = queue.pop(0)
        for nr, nc in neighbors(r, c):
            if (r, c) <= (nr, nc):
                edge = ((r, c), (nr, nc))
            else:
                edge = ((nr, nc), (r, c))
            if edge not in visited_edges:
                visited_edges.add(edge)
                if (nr, nc) not in visited_vertices:
                    visited_vertices.add((nr, nc))
                    queue.append((nr, nc))

    if len(visited_edges) != edge_count:
        raise ValueError("ループが複数存在する可能性があります")

    # ハード制約 H-7: ループ長下限チェック
    # 辺の数が 2 * (rows + cols) 未満なら盤面外周だけをなぞる短すぎるループとみなす
    if edge_count < 2 * (size.rows + size.cols):
        raise ValueError("ループ長がハード制約を満たしていません")

    # ヒント数字の整合をチェック
    clues = puzzle.get("clues")
    if not isinstance(clues, list):
        raise ValueError("clues フィールドが存在しません")
    calculated = _calculate_clues(edges, size)
    if clues != calculated:
        raise ValueError("clues が solutionEdges と一致しません")

    # ハード制約 H-8: 0 の隣接禁止をチェック
    for r in range(size.rows):
        for c in range(size.cols):
            if clues[r][c] == 0:
                if r + 1 < size.rows and clues[r + 1][c] == 0:
                    raise ValueError("0 が縦に隣接しています")
                if c + 1 < size.cols and clues[r][c + 1] == 0:
                    raise ValueError("0 が横に隣接しています")

    # ハード制約 H-9: 線カーブ比率下限チェック
    curve_ratio = _calculate_curve_ratio(edges, size)
    if curve_ratio < 0.15:
        raise ValueError("線カーブ比率がハード制約を満たしていません")


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
                    line += f" {clues[row_idx][c]} "
            lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    # 実行例：生成したパズルを保存
    pzl = generate_puzzle(4, 4, difficulty="easy")
    path = save_puzzle(pzl)
    print(f"{path} を作成しました")
    # 生成した盤面を ASCII で表示
    print(puzzle_to_ascii(pzl))
