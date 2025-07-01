"""簡易的なスリザーリンク盤面生成モジュール"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import random
from typing import Any, Dict, List


@dataclass
class PuzzleSize:
    """盤面サイズを表すデータクラス"""

    rows: int
    cols: int


Puzzle = Dict[str, Any]


def _create_empty_edges(size: PuzzleSize) -> Dict[str, List[List[bool]]]:
    """solutionEdges フィールド用の空の二次元配列を作成する"""
    horizontal = [[False for _ in range(size.cols)] for _ in range(size.rows + 1)]
    vertical = [[False for _ in range(size.cols + 1)] for _ in range(size.rows)]
    return {"horizontal": horizontal, "vertical": vertical}


def _add_outer_loop(edges: Dict[str, List[List[bool]]], size: PuzzleSize) -> None:
    """盤面外周に沿った単純なループを追加する"""
    # 上辺と下辺
    for c in range(size.cols):
        edges["horizontal"][0][c] = True
        edges["horizontal"][size.rows][c] = True
    # 左辺と右辺
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


ALLOWED_DIFFICULTIES = {"easy", "normal", "hard", "expert"}


def generate_puzzle(
    rows: int, cols: int, difficulty: str = "normal", *, seed: int | None = None
) -> Puzzle:
    """簡易な盤面を生成して返す

    :param rows: 盤面の行数
    :param cols: 盤面の列数
    :param difficulty: 難易度ラベル
    :param seed: 乱数シード。再現したいときに指定する
    """

    if difficulty not in ALLOWED_DIFFICULTIES:
        raise ValueError(f"difficulty は {ALLOWED_DIFFICULTIES} のいずれかで指定")

    if seed is not None:
        random.seed(seed)

    size = PuzzleSize(rows=rows, cols=cols)
    edges = _create_empty_edges(size)
    _add_outer_loop(edges, size)
    clues = _calculate_clues(edges, size)

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    puzzle: Puzzle = {
        "id": f"sl_{rows}x{cols}_{difficulty}_{timestamp}",
        "size": {"rows": rows, "cols": cols},
        "clues": clues,
        "solutionEdges": edges,
        "difficulty": difficulty,
        "createdBy": "auto-gen-v1",
        "createdAt": datetime.utcnow().date().isoformat(),
    }
    return puzzle


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
    return puzzles


def save_puzzles(puzzles: List[Puzzle], directory: str | Path = "data") -> Path:
    """複数のパズルをひとつの JSON ファイルに保存する"""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "map_gridtrace.json"
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(puzzles, fp, ensure_ascii=False, indent=2)
    return file_path


def validate_puzzle(puzzle: Puzzle) -> None:
    """パズルデータの整合性を簡易チェックする関数"""

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

    # ヒント数字の整合をチェック
    clues = puzzle.get("clues")
    if not isinstance(clues, list):
        raise ValueError("clues フィールドが存在しません")
    calculated = _calculate_clues(edges, size)
    if clues != calculated:
        raise ValueError("clues が solutionEdges と一致しません")


if __name__ == "__main__":
    # 実行例：生成したパズルを保存
    pzl = generate_puzzle(4, 4, difficulty="easy")
    path = save_puzzle(pzl)
    print(f"{path} を作成しました")
