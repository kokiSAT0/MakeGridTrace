"""難易度ごとのパズルをまとめて生成するスクリプト"""

from __future__ import annotations

import argparse

from .generator import generate_multiple_puzzles, save_puzzles


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
    args = parser.parse_args()

    puzzles = generate_multiple_puzzles(args.rows, args.cols, args.count_each)
    path = save_puzzles(puzzles)
    print(f"{path} を作成しました")


if __name__ == "__main__":
    main()
