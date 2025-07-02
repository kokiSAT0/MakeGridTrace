"""共通定数や簡易ヘルパー関数を定義するモジュール"""

from __future__ import annotations

# ソルバーが探索する最大ステップ数
# ステップ数を増やすと解の探索精度が上がるが時間もかかる
MAX_SOLVER_STEPS = 500000


def _evaluate_difficulty(steps: int, depth: int) -> str:
    """ソルバー統計から難易度を推定する関数"""

    # 解析に用いたステップ数とバックトラック深さから判断する
    if steps < 1000 and depth <= 2:
        return "easy"
    if steps < 10000 and depth <= 10:
        return "normal"
    if steps < 100000 and depth <= 30:
        return "hard"
    return "expert"


__all__ = ["MAX_SOLVER_STEPS", "_evaluate_difficulty"]
