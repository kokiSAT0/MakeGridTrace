"""パズルを保存・読み込みする処理をまとめたモジュール"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

# generator への循環参照を避けるため型エイリアスだけ定義
Puzzle = Dict[str, Any]


def save_puzzle(puzzle: Puzzle, directory: str | Path = "data") -> Path:
    """単一のパズルを JSON 形式で保存する"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "map_gridtrace.json"
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(puzzle, fp, ensure_ascii=False, indent=2)
    return file_path


def save_puzzles(puzzles: List[Puzzle], directory: str | Path = "data") -> Path:
    """複数パズルをまとめて JSON 保存する"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "map_gridtrace.json"
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(puzzles, fp, ensure_ascii=False, indent=2)
    return file_path


__all__ = ["save_puzzle", "save_puzzles"]
