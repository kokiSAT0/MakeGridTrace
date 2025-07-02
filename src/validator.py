"""パズルデータ検証モジュール"""

from __future__ import annotations

from typing import Dict, Any, List
from .solver import PuzzleSize, calculate_clues

Puzzle = Dict[str, Any]


def _calculate_curve_ratio(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> float:
    """ループ中の曲がり角割合を計算する"""

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
    total = sum(sum(row) for row in edges["horizontal"]) + sum(
        sum(row) for row in edges["vertical"]
    )
    return curve_count / total if total > 0 else 0.0


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

    if edge_count < 2 * (size.rows + size.cols):
        raise ValueError("ループ長がハード制約を満たしていません")

    clues_full = puzzle.get("cluesFull")
    if not isinstance(clues_full, list):
        raise ValueError("cluesFull フィールドが存在しません")
    calculated = calculate_clues(edges, size)
    if clues_full != calculated:
        raise ValueError("cluesFull が solutionEdges と一致しません")

    clues = puzzle.get("clues")
    if not isinstance(clues, list):
        raise ValueError("clues フィールドが存在しません")
    for r in range(size.rows):
        for c in range(size.cols):
            val = clues[r][c]
            if val is not None and val != clues_full[r][c]:
                raise ValueError("clues が cluesFull と一致しません")

    for r in range(size.rows):
        for c in range(size.cols):
            if clues_full[r][c] == 0:
                if r + 1 < size.rows and clues_full[r + 1][c] == 0:
                    raise ValueError("0 が縦に隣接しています")
                if c + 1 < size.cols and clues_full[r][c + 1] == 0:
                    raise ValueError("0 が横に隣接しています")

    curve_ratio = _calculate_curve_ratio(edges, size)
    if curve_ratio < 0.15:
        raise ValueError("線カーブ比率がハード制約を満たしていません")
