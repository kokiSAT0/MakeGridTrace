"""generator や入出力モジュールの関数を公開するパッケージ用モジュール"""

from importlib import import_module
from typing import Any

__all__ = [
    "generate_puzzle",
    "generate_puzzle_parallel",
    "save_puzzle",
    "validate_puzzle",
    "generate_multiple_puzzles",
    "save_puzzles",
    "puzzle_to_ascii",
    "build_quality_histogram",
    "generate_loop",
]


def __getattr__(name: str) -> Any:
    """必要になったタイミングで対象モジュールを読み込む"""

    if name in {
        "generate_puzzle",
        "generate_multiple_puzzles",
        "puzzle_to_ascii",
        "generate_puzzle_parallel",
    }:
        module = import_module(".generator", __name__)
        return getattr(module, name)

    if name == "generate_loop":
        module = import_module(".loop_wilson", __name__)
        return getattr(module, name)

    if name in {"save_puzzle", "save_puzzles"}:
        module = import_module(".puzzle_io", __name__)
        return getattr(module, name)

    if name == "build_quality_histogram":
        module = import_module(".quality_histogram", __name__)
        return getattr(module, name)

    if name == "validate_puzzle":
        module = import_module(".validator", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name}")
