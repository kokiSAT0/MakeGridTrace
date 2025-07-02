"""難易度ごとのパズルをまとめて生成するスクリプト"""

from __future__ import annotations

import logging

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.generator import generate_multiple_puzzles, puzzle_to_ascii, setup_logging
    from src.puzzle_io import save_puzzles
else:
    try:
        # パッケージ実行時は相対インポート
        from .generator import generate_multiple_puzzles, puzzle_to_ascii, setup_logging
        from .puzzle_io import save_puzzles
    except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
        # スクリプトとして直接実行されたときは同じディレクトリからインポートする
        from generator import generate_multiple_puzzles, puzzle_to_ascii, setup_logging
        from puzzle_io import save_puzzles


# コマンドラインから実行される関数
def main() -> None:
    """引数を解釈してパズルを生成し保存する"""

    parser = argparse.ArgumentParser(
        description="easy/normal/hard/expert を同数生成して保存します"
    )
    parser.add_argument("rows", type=int, help="盤面の行数")
    parser.add_argument("cols", type=int, help="盤面の列数")
    parser.add_argument(
        "count_each", type=int, default=1, help="各難易度の生成数 (デフォルト:1)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="並列生成プロセス数",
    )
    args = parser.parse_args()

    puzzles = generate_multiple_puzzles(
        args.rows,
        args.cols,
        args.count_each,
        jobs=args.jobs,
        worker_log_level=logging.WARNING if args.jobs > 1 else logging.INFO,
    )
    path = save_puzzles(puzzles)
    print(f"{path} を作成しました")
    # 生成した各パズルを ASCII で表示
    for pzl in puzzles:
        print(f"--- {pzl['difficulty']} ---")
        print(puzzle_to_ascii(pzl))


if __name__ == "__main__":
    # ログ設定を行ってからメイン処理を呼び出す
    setup_logging()
    main()
