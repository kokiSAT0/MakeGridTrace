"""簡易的なスリザーリンク盤面生成モジュール"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import List, Dict, Any


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


def generate_puzzle(rows: int, cols: int) -> Puzzle:
    """簡易な盤面を生成して返す"""
    size = PuzzleSize(rows=rows, cols=cols)
    edges = _create_empty_edges(size)
    _add_outer_loop(edges, size)
    clues = _calculate_clues(edges, size)
    puzzle: Puzzle = {
        "id": f"sl_{rows}x{cols}_sample_{int(datetime.utcnow().timestamp())}",
        "size": {"rows": rows, "cols": cols},
        "clues": clues,
        "solutionEdges": edges,
        "difficulty": "easy",
        "createdBy": "auto-gen-v1",
        "createdAt": datetime.utcnow().date().isoformat(),
    }
    return puzzle


if __name__ == "__main__":
    # 実行例：生成したパズルを JSON 形式で保存
    pzl = generate_puzzle(4, 4)
    with open("data/sample.json", "w", encoding="utf-8") as f:
        json.dump(pzl, f, ensure_ascii=False, indent=2)
    print("sample.json を作成しました")
