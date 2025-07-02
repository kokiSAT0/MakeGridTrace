"""共通で使う型エイリアスをまとめたモジュール"""

from typing import Any, Dict

# Puzzle データを表す辞書型。キーは文字列で値は任意の型を許容
Puzzle = Dict[str, Any]

__all__ = ["Puzzle"]
