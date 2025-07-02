"""共通で使用する型エイリアスを定義するモジュール"""

from typing import Any, Dict

# Puzzle は文字列キーを持つ辞書型とする
Puzzle = Dict[str, Any]

__all__ = ["Puzzle"]
