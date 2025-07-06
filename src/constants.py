"""共通定数や簡易ヘルパー関数を定義するモジュール"""

from __future__ import annotations

# 従来のデフォルト値を参考として残しておく
# 現在は関数側で盤面サイズから計算するため、この値は直接は使わない
DEFAULT_SOLVER_STEP_LIMIT = 500000


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


__all__ = ["DEFAULT_SOLVER_STEP_LIMIT", "_evaluate_difficulty"]
