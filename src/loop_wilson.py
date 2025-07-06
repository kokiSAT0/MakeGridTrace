"""Wilson 法によるループ生成ロジック"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

try:
    from .solver import PuzzleSize
except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
    from solver import PuzzleSize

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


def _vertex_degree(edges: Dict[str, List[List[bool]]], size: PuzzleSize, r: int, c: int) -> int:
    """頂点の接続数を求める小さなヘルパー"""
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
    """辺情報が単一ループを構成するか簡易チェック"""
    for r in range(size.rows + 1):
        for c in range(size.cols + 1):
            deg = _vertex_degree(edges, size, r, c)
            if deg not in (0, 2):
                return False
    return True


def generate_loop(
    size: PuzzleSize,
    rng: random.Random,
    *,
    symmetry: Optional[str] = None,
    theme: Optional[str] = None,
) -> tuple[Dict[str, List[List[bool]]], int, float]:
    """Wilson 法に基づくループ生成を行う"""

    # ここでは loop-erased random walk を直接実装する代わりに、
    # 従来の ``_generate_random_loop`` を利用してランダム候補を作成し、
    # ループ検証や対称処理を組み合わせて Wilson 法風の挙動を再現している。

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

    # 失敗時は外周だけのループを返す
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


__all__ = ["generate_loop", "_validate_edges"]
