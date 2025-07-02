"""ループ生成に関する補助関数をまとめたモジュール"""

from __future__ import annotations

from typing import Dict, List
import random

from .solver import PuzzleSize


def _create_empty_edges(size: PuzzleSize) -> Dict[str, List[List[bool]]]:
    """solutionEdges 用の空の二次元配列を作成する"""
    horizontal = [[False for _ in range(size.cols)] for _ in range(size.rows + 1)]
    vertical = [[False for _ in range(size.cols + 1)] for _ in range(size.rows)]
    return {"horizontal": horizontal, "vertical": vertical}


def _generate_random_loop(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize, rng: random.Random
) -> None:
    """バックトラックでランダムなループを生成する

    :param rng: 乱数生成に利用する ``random.Random`` インスタンス
    """

    degrees = [[0 for _ in range(size.cols + 1)] for _ in range(size.rows + 1)]

    def edge_exists(a: tuple[int, int], b: tuple[int, int]) -> bool:
        """2 点間の辺が存在するか確認"""
        if a[0] == b[0]:
            r = a[0]
            c = min(a[1], b[1])
            return edges["horizontal"][r][c]
        else:
            c = a[1]
            r = min(a[0], b[0])
            return edges["vertical"][r][c]

    def add_edge(a: tuple[int, int], b: tuple[int, int]) -> None:
        """2 点を結ぶ辺を追加する"""
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
        """2 点を結ぶ辺を削除する"""
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

    min_len = max(2 * (size.rows + size.cols), 4)

    start = (rng.randint(0, size.rows), rng.randint(0, size.cols))
    path: list[tuple[int, int]] = [start]

    def dfs(current: tuple[int, int]) -> bool:
        if len(path) >= min_len and current == start and len(path) > 1:
            return True
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        rng.shuffle(directions)
        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr <= size.rows and 0 <= nc <= size.cols):
                continue
            nxt = (nr, nc)
            if degrees[nr][nc] >= 2 or degrees[current[0]][current[1]] >= 2:
                continue
            if edge_exists(current, nxt):
                continue
            if nxt == start and len(path) + 1 < min_len:
                continue
            add_edge(current, nxt)
            path.append(nxt)
            if dfs(nxt):
                return True
            path.pop()
            remove_edge(current, nxt)
        return False

    if not dfs(start):
        # 失敗したら外周ループを生成
        for c in range(size.cols):
            edges["horizontal"][0][c] = True
            edges["horizontal"][size.rows][c] = True
        for r in range(size.rows):
            edges["vertical"][r][0] = True
            edges["vertical"][r][size.cols] = True


def _apply_rotational_symmetry(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> None:
    """180 度回転対称になるよう辺情報を補正"""
    horizontal = edges["horizontal"]
    for r in range(size.rows + 1):
        for c in range(size.cols):
            sr = size.rows - r
            sc = size.cols - c - 1
            val = horizontal[r][c] or horizontal[sr][sc]
            horizontal[r][c] = val
            horizontal[sr][sc] = val
    vertical = edges["vertical"]
    for r in range(size.rows):
        for c in range(size.cols + 1):
            sr = size.rows - r - 1
            sc = size.cols - c
            val = vertical[r][c] or vertical[sr][sc]
            vertical[r][c] = val
            vertical[sr][sc] = val


def _apply_vertical_symmetry(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> None:
    """上下方向で鏡映対称になるよう辺情報を補正"""

    horizontal = edges["horizontal"]
    for r in range(size.rows + 1):
        sr = size.rows - r
        for c in range(size.cols):
            val = horizontal[r][c] or horizontal[sr][c]
            horizontal[r][c] = val
            horizontal[sr][c] = val

    vertical = edges["vertical"]
    for r in range(size.rows):
        sr = size.rows - r
        for c in range(size.cols + 1):
            val = vertical[r][c] or vertical[sr - 1][c]
            vertical[r][c] = val
            vertical[sr - 1][c] = val


def _apply_horizontal_symmetry(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> None:
    """左右方向で鏡映対称になるよう辺情報を補正"""

    horizontal = edges["horizontal"]
    for r in range(size.rows + 1):
        for c in range(size.cols):
            sc = size.cols - c - 1
            val = horizontal[r][c] or horizontal[r][sc]
            horizontal[r][c] = val
            horizontal[r][sc] = val

    vertical = edges["vertical"]
    for r in range(size.rows):
        for c in range(size.cols + 1):
            sc = size.cols - c
            val = vertical[r][c] or vertical[r][sc]
            vertical[r][c] = val
            vertical[r][sc] = val


def _count_edges(edges: Dict[str, List[List[bool]]]) -> int:
    """True の数を数えてループ長を求める"""
    return sum(sum(row) for row in edges["horizontal"]) + sum(
        sum(row) for row in edges["vertical"]
    )


def _calculate_curve_ratio(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> float:
    """ループ中の曲がり角割合を計算"""
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


__all__ = [
    "_create_empty_edges",
    "_generate_random_loop",
    "_apply_rotational_symmetry",
    "_apply_vertical_symmetry",
    "_apply_horizontal_symmetry",
    "_count_edges",
    "_calculate_curve_ratio",
]
