import random
import time
from typing import Optional

from . import generator


def run(rows: int, cols: int, n: int = 1, *, seed: Optional[int] = None) -> float:
    """指定回数パズルを生成して平均時間を返す簡易ベンチマーク関数"""
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n):
        start = time.perf_counter()
        generator.generate_puzzle(rows, cols, seed=rng.randint(0, 2**32))
        total += time.perf_counter() - start
    avg = total / n if n else 0.0
    print(f"平均生成時間: {avg:.3f} 秒")
    return avg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="パズル生成ベンチマーク")
    parser.add_argument("rows", type=int, help="盤面の行数")
    parser.add_argument("cols", type=int, help="盤面の列数")
    parser.add_argument("-n", type=int, default=1, help="生成回数")
    parser.add_argument("--seed", type=int, help="乱数シード")
    args = parser.parse_args()
    run(args.rows, args.cols, args.n, seed=args.seed)
