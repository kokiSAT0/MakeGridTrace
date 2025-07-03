"""小サイズのループパターンを定義するモジュール"""

from __future__ import annotations

from typing import Dict, List


def _make_square_pattern(size: int) -> Dict[str, List[List[bool]]]:
    """正方形の外周だけを通る単純なループを作成する"""
    horizontal = [[False for _ in range(size)] for _ in range(size + 1)]
    vertical = [[False for _ in range(size + 1)] for _ in range(size)]
    for c in range(size):
        horizontal[0][c] = True
        horizontal[size][c] = True
    for r in range(size):
        vertical[r][0] = True
        vertical[r][size] = True
    return {"horizontal": horizontal, "vertical": vertical}


# 実際に利用するパターンを辞書にまとめる
PATTERNS: Dict[int, Dict[str, List[List[bool]]]] = {
    3: _make_square_pattern(3),
    4: _make_square_pattern(4),
}

__all__ = ["PATTERNS"]
