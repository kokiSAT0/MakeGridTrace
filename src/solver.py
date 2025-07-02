# スリザーリンク用簡易ソルバーモジュール

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class PuzzleSize:
    """盤面サイズを表すデータクラス"""

    rows: int
    cols: int


@dataclass
class Edge:
    """盤面上の1本の線分を表すデータクラス"""

    orientation: str  # 'h' または 'v'
    r: int
    c: int
    vertices: Tuple[Tuple[int, int], Tuple[int, int]]
    cells: List[Tuple[int, int]]


@dataclass
class Board:
    """解析中の盤面状態を保持するデータクラス"""

    size: PuzzleSize
    edges: List[Edge]
    edge_state: List[bool]
    vertex_degree: List[List[int]]
    cell_count: List[List[int]]
    cell_unknown: List[List[int]]


def _create_edges(size: PuzzleSize) -> List[Edge]:
    """盤面サイズから Edge の一覧を生成する"""
    edges: List[Edge] = []
    for r in range(size.rows + 1):
        for c in range(size.cols):
            cells = []
            if r < size.rows:
                cells.append((r, c))
            if r > 0:
                cells.append((r - 1, c))
            edge = Edge(
                orientation="h",
                r=r,
                c=c,
                vertices=((r, c), (r, c + 1)),
                cells=cells,
            )
            edges.append(edge)
    for r in range(size.rows):
        for c in range(size.cols + 1):
            cells = []
            if c < size.cols:
                cells.append((r, c))
            if c > 0:
                cells.append((r, c - 1))
            edge = Edge(
                orientation="v",
                r=r,
                c=c,
                vertices=((r, c), (r + 1, c)),
                cells=cells,
            )
            edges.append(edge)
    return edges


def _init_board(size: PuzzleSize) -> Board:
    """Board オブジェクトを初期化して返す"""
    edges = _create_edges(size)
    n = len(edges)
    edge_state = [False] * n
    vertex_degree = [[0 for _ in range(size.cols + 1)] for _ in range(size.rows + 1)]
    cell_count = [[0 for _ in range(size.cols)] for _ in range(size.rows)]
    cell_unknown = [[4 for _ in range(size.cols)] for _ in range(size.rows)]
    return Board(size, edges, edge_state, vertex_degree, cell_count, cell_unknown)


def calculate_clues(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> List[List[int]]:
    """solutionEdges からヒント数字を計算する"""
    clues = [[0 for _ in range(size.cols)] for _ in range(size.rows)]
    horizontal = edges["horizontal"]
    vertical = edges["vertical"]
    for r in range(size.rows):
        for c in range(size.cols):
            count = 0
            if horizontal[r][c]:
                count += 1
            if horizontal[r + 1][c]:
                count += 1
            if vertical[r][c]:
                count += 1
            if vertical[r][c + 1]:
                count += 1
            clues[r][c] = count
    return clues


# 外部から呼び出す関数名は count_solutions とする


def count_solutions(
    clues: List[List[int | None]],
    size: PuzzleSize,
    *,
    limit: int = 2,
    return_stats: bool = False,
    step_limit: int | None = None,
) -> int | tuple[int, Dict[str, int]]:
    """バックトラックで解の個数を数える簡易ソルバー"""

    if step_limit is None and return_stats:
        step_limit = 500000  # 解析ステップ数の上限

    board = _init_board(size)
    n = len(board.edges)
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
            # すべての辺の状態が決まったので条件を確認
            for r in range(size.rows + 1):
                for c in range(size.cols + 1):
                    if board.vertex_degree[r][c] not in (0, 2):
                        return
            for r in range(size.rows):
                for c in range(size.cols):
                    clue = clues[r][c]
                    if clue is not None and board.cell_count[r][c] != clue:
                        return
            horizontal = [
                [False for _ in range(size.cols)] for _ in range(size.rows + 1)
            ]
            vertical = [[False for _ in range(size.cols + 1)] for _ in range(size.rows)]
            for i, e in enumerate(board.edges):
                if not board.edge_state[i]:
                    continue
                if e.orientation == "h":
                    horizontal[e.r][e.c] = True
                else:
                    vertical[e.r][e.c] = True
            # 検証のために cluesFull を含むパズルオブジェクトを作成
            from . import generator as _gen  # 循環インポート回避のため

            clues_full = calculate_clues(
                {"horizontal": horizontal, "vertical": vertical}, size
            )
            puzzle = {
                "size": {"rows": size.rows, "cols": size.cols},
                "solutionEdges": {"horizontal": horizontal, "vertical": vertical},
                "clues": clues,
                "cluesFull": clues_full,
            }
            try:
                _gen.validate_puzzle(puzzle)
            except ValueError:
                return
            solutions += 1
            return

        edge = board.edges[idx]
        for val in (False, True):
            board.edge_state[idx] = val
            for vr, vc in edge.vertices:
                board.vertex_degree[vr][vc] += int(val)
            for cr, cc in edge.cells:
                board.cell_unknown[cr][cc] -= 1
                if val:
                    board.cell_count[cr][cc] += 1
            # 部分的な矛盾をチェック
            ok = True
            for vr, vc in edge.vertices:
                if board.vertex_degree[vr][vc] > 2:
                    ok = False
                    break
            if ok:
                for cr, cc in edge.cells:
                    clue = clues[cr][cc]
                    if clue is None:
                        continue
                    used = board.cell_count[cr][cc]
                    unknown = board.cell_unknown[cr][cc]
                    if used > clue or used + unknown < clue:
                        ok = False
                        break
            if ok:
                dfs(idx + 1, depth + 1)
            for vr, vc in edge.vertices:
                board.vertex_degree[vr][vc] -= int(val)
            for cr, cc in edge.cells:
                if val:
                    board.cell_count[cr][cc] -= 1
                board.cell_unknown[cr][cc] += 1
        board.edge_state[idx] = False

    dfs(0, 0)
    if return_stats:
        return solutions, {"steps": steps, "max_depth": max_depth}
    return solutions


__all__ = ["PuzzleSize", "Edge", "Board", "calculate_clues", "count_solutions"]
