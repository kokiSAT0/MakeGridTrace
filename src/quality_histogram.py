"""Quality Score の分布を計算するモジュール"""

from __future__ import annotations

import random
from typing import List, Tuple

from .generator import generate_puzzle


def build_quality_histogram(
    rows: int,
    cols: int,
    samples: int,
    *,
    seed: int | None = None,
) -> Tuple[List[int], float]:
    """複数パズルを生成して Quality Score の統計を求める関数

    ``samples`` 個のパズルを生成し、それぞれの Quality Score を計算します。
    スコアを 0--100 の範囲で 5 点刻みの 21 区間に集計し、
    95 パーセンタイル (全体の上位 5% 境界値) を返します。

    :param rows: 盤面の行数
    :param cols: 盤面の列数
    :param samples: 生成するパズル数
    :param seed: 乱数シード。指定しない場合はランダム
    :return: (ヒストグラム配列, P95 値)
    """

    if samples <= 0:
        raise ValueError("samples は 1 以上を指定してください")

    rng = random.Random(seed)
    scores: List[float] = []
    for _ in range(samples):
        puzzle = generate_puzzle(rows, cols, seed=rng.randint(0, 2**32 - 1))
        scores.append(float(puzzle["qualityScore"]))

    scores.sort()

    histogram = [0 for _ in range(21)]
    for score in scores:
        idx = min(20, int(score // 5))
        histogram[idx] += 1

    if scores:
        index = max(0, int(len(scores) * 0.95) - 1)
        p95 = scores[index]
    else:
        p95 = 0.0

    return histogram, p95


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(
        description="Quality Score のヒストグラムと P95 を計算します"
    )
    parser.add_argument("rows", type=int, help="盤面の行数")
    parser.add_argument("cols", type=int, help="盤面の列数")
    parser.add_argument("--samples", type=int, default=10, help="生成数")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード")
    args = parser.parse_args()

    hist, p95 = build_quality_histogram(
        args.rows, args.cols, args.samples, seed=args.seed
    )
    pprint(hist)
    print(f"P95 = {p95:.2f}")
