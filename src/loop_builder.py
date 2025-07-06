"""ループ生成に関する補助関数をまとめたモジュール"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING
import random

import numpy as np
from numba import njit

try:
    # 定義済みの小ループパターンを読み込む
    from .pattern_builder import PATTERNS
except Exception:  # pragma: no cover - スクリプト実行時のフォールバック
    from pattern_builder import PATTERNS  # type: ignore

if TYPE_CHECKING:
    from src.solver import PuzzleSize
else:
    try:
        # パッケージとして実行された場合の相対インポート
        from .solver import PuzzleSize
    except ImportError:  # pragma: no cover - スクリプト実行時のフォールバック
        # スクリプトとして直接実行されたときは同じディレクトリからインポートする
        from solver import PuzzleSize


def _create_empty_edges(size: PuzzleSize) -> Dict[str, List[List[bool]]]:
    """solutionEdges 用の空の二次元配列を作成する"""
    horizontal = [[False for _ in range(size.cols)] for _ in range(size.rows + 1)]
    vertical = [[False for _ in range(size.cols + 1)] for _ in range(size.rows)]
    return {"horizontal": horizontal, "vertical": vertical}


def _pack_edges(edges: Dict[str, List[List[bool]]]) -> tuple[np.ndarray, np.ndarray]:
    """辺情報を NumPy 配列へ変換する簡易ヘルパー"""

    # ブール値を 0/1 の uint8 配列へ変換する
    h = np.array(edges["horizontal"], dtype=np.uint8)
    v = np.array(edges["vertical"], dtype=np.uint8)
    return h, v


def _generate_random_loop(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize, rng: random.Random
) -> None:
    """深さ優先探索でランダムなループを作成する関数

    盤面上をランダムに進みながらループを構築します。探索中は各頂点
    （マス目の角）での接続数 ``degrees`` を管理し、2 本以上の線が
    つながらないようにします。最終的に閉じたループができれば成功です。

    :param rng: 乱数生成に利用する ``random.Random`` インスタンス
    """

    # 辺情報を NumPy 配列へ変換し、1/0 のビットボードとして扱う
    h_edges = np.array(edges["horizontal"], dtype=np.uint8)
    v_edges = np.array(edges["vertical"], dtype=np.uint8)

    # 各頂点に接続する辺の数を記録する二次元配列
    # 各頂点の次数を NumPy 配列で管理する
    degrees = np.zeros((size.rows + 1, size.cols + 1), dtype=np.uint8)

    def add_edge(a: tuple[int, int], b: tuple[int, int]) -> None:
        """2 点を結ぶ辺を追加し ``degrees`` を更新"""
        if a[0] == b[0]:
            r = a[0]
            c = min(a[1], b[1])
            h_edges[r, c] = 1
        else:
            c = a[1]
            r = min(a[0], b[0])
            v_edges[r, c] = 1
        degrees[a[0]][a[1]] += 1
        degrees[b[0]][b[1]] += 1

    def remove_edge(a: tuple[int, int], b: tuple[int, int]) -> None:
        """2 点を結ぶ辺を削除し ``degrees`` を更新"""
        if a[0] == b[0]:
            r = a[0]
            c = min(a[1], b[1])
            h_edges[r, c] = 0
        else:
            c = a[1]
            r = min(a[0], b[0])
            v_edges[r, c] = 0
        degrees[a[0]][a[1]] -= 1
        degrees[b[0]][b[1]] -= 1

    # 盤面の周囲長の2倍を下回らないようループの最小長を設定
    min_len = max(2 * (size.rows + size.cols), 4)
    # 探索の上限長を周囲長の 1.5 倍にして無限ループを防ぐ
    max_len = int(1.5 * (size.rows + size.cols) * 2)

    # 探索開始点をランダムに決め、現在の経路 ``route`` に追加
    start_vertex = (rng.randint(0, size.rows), rng.randint(0, size.cols))
    route: list[tuple[int, int]] = [start_vertex]

    # 事前に各頂点ごとの移動方向の順序を乱数で決めておく
    # 繰り返し ``random.shuffle`` を呼ばずに済むため高速化できる
    base_dirs = np.array([(0, 1), (1, 0), (0, -1), (-1, 0)], dtype=np.int8)
    np_rng = np.random.default_rng(rng.randint(0, 2**32))
    vertex_dirs = {
        (r, c): [tuple(v) for v in np_rng.permutation(base_dirs)]
        for r in range(size.rows + 1)
        for c in range(size.cols + 1)
    }

    def _search_loop_dfs(current: tuple[int, int]) -> bool:
        """深さ優先で経路を探索し閉ループを探す"""

        # 最小長を満たした状態で開始点に戻れば成功
        if len(route) >= min_len and current == start_vertex and len(route) > 1:
            return True

        # 設定した上限を超えたら失敗とする
        if len(route) > max_len:
            return False

        # 現在位置に対応する方向順序をそのまま試す
        for dr, dc in vertex_dirs[current]:
            nr, nc = current[0] + dr, current[1] + dc
            # 盤面外に出る候補は除外
            if not (0 <= nr <= size.rows and 0 <= nc <= size.cols):
                continue
            next_vertex = (nr, nc)
            # 各頂点の次数が2を超える場合は無効
            if degrees[nr][nc] >= 2 or degrees[current[0]][current[1]] >= 2:
                continue
            # すでに同じ辺が存在する場合もスキップ
            if current[0] == next_vertex[0]:
                r = current[0]
                c = min(current[1], next_vertex[1])
                if h_edges[r, c] != 0:
                    continue
            else:
                c = current[1]
                r = min(current[0], next_vertex[0])
                if v_edges[r, c] != 0:
                    continue
            # 開始点へ戻る際は最小長を満たしているか確認
            if next_vertex == start_vertex and len(route) + 1 < min_len:
                continue

            # 候補辺を追加して再帰呼び出し
            add_edge(current, next_vertex)
            route.append(next_vertex)
            if _search_loop_dfs(next_vertex):
                return True
            # 失敗した場合は追加した辺を取り消す
            route.pop()
            remove_edge(current, next_vertex)
        return False

    def _search_loop_bfs() -> bool:
        """幅優先でループを探索する簡易実装"""

        from collections import deque

        State = tuple[
            tuple[int, int],
            list[tuple[int, int]],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]

        start_state: State = (
            start_vertex,
            [start_vertex],
            h_edges.copy(),
            v_edges.copy(),
            degrees.copy(),
        )

        queue: deque[State] = deque([start_state])
        steps = 0
        while queue and steps < 2000:
            current, rpath, h, v, deg = queue.popleft()
            steps += 1
            if len(rpath) >= min_len and current == start_vertex and len(rpath) > 1:
                h_edges[:, :] = h
                v_edges[:, :] = v
                degrees[:, :] = deg
                route[:] = rpath
                return True

            if len(rpath) > max_len:
                continue

            # BFS でも頂点ごとに決めた方向順序を使用
            for dr, dc in vertex_dirs[current]:
                nr, nc = current[0] + dr, current[1] + dc
                if not (0 <= nr <= size.rows and 0 <= nc <= size.cols):
                    continue
                if deg[nr, nc] >= 2 or deg[current[0], current[1]] >= 2:
                    continue
                if current[0] == nr:
                    r_idx = current[0]
                    c_idx = min(current[1], nc)
                    if h[r_idx, c_idx] != 0:
                        continue
                else:
                    c_idx = current[1]
                    r_idx = min(current[0], nr)
                    if v[r_idx, c_idx] != 0:
                        continue
                if (nr, nc) == start_vertex and len(rpath) + 1 < min_len:
                    continue

                h2 = h.copy()
                v2 = v.copy()
                deg2 = deg.copy()
                if current[0] == nr:
                    h2[r_idx, c_idx] = 1
                else:
                    v2[r_idx, c_idx] = 1
                deg2[current[0], current[1]] += 1
                deg2[nr, nc] += 1
                queue.append(((nr, nc), rpath + [(nr, nc)], h2, v2, deg2))

        return False

    # まず DFS で探索し、失敗したら BFS を試す
    if not _search_loop_dfs(start_vertex) and not _search_loop_bfs():
        for c in range(size.cols):
            h_edges[0, c] = 1
            h_edges[size.rows, c] = 1
        for r in range(size.rows):
            v_edges[r, 0] = 1
            v_edges[r, size.cols] = 1

    # 生成した NumPy 配列を Python のリストへ戻す
    edges["horizontal"] = h_edges.astype(bool).tolist()
    edges["vertical"] = v_edges.astype(bool).tolist()


def _apply_rotational_symmetry(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> None:
    """180 度回転対称になるよう辺情報を補正"""
    horizontal = edges["horizontal"]
    for r in range(size.rows + 1):
        for c in range(size.cols):
            sr = size.rows - r
            sc = size.cols - c - 1
            val = horizontal[r][c] or horizontal[sr][sc]
            horizontal[r][c] = val
            horizontal[sr][sc] = val
    vertical = edges["vertical"]
    for r in range(size.rows):
        for c in range(size.cols + 1):
            sr = size.rows - r - 1
            sc = size.cols - c
            val = vertical[r][c] or vertical[sr][sc]
            vertical[r][c] = val
            vertical[sr][sc] = val


def _apply_vertical_symmetry(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> None:
    """上下方向で鏡映対称になるよう辺情報を補正"""

    horizontal = edges["horizontal"]
    for r in range(size.rows + 1):
        sr = size.rows - r
        for c in range(size.cols):
            val = horizontal[r][c] or horizontal[sr][c]
            horizontal[r][c] = val
            horizontal[sr][c] = val

    vertical = edges["vertical"]
    for r in range(size.rows):
        sr = size.rows - r
        for c in range(size.cols + 1):
            val = vertical[r][c] or vertical[sr - 1][c]
            vertical[r][c] = val
            vertical[sr - 1][c] = val


def _apply_horizontal_symmetry(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> None:
    """左右方向で鏡映対称になるよう辺情報を補正"""

    horizontal = edges["horizontal"]
    for r in range(size.rows + 1):
        for c in range(size.cols):
            sc = size.cols - c - 1
            val = horizontal[r][c] or horizontal[r][sc]
            horizontal[r][c] = val
            horizontal[r][sc] = val

    vertical = edges["vertical"]
    for r in range(size.rows):
        for c in range(size.cols + 1):
            sc = size.cols - c
            val = vertical[r][c] or vertical[r][sc]
            vertical[r][c] = val
            vertical[r][sc] = val


@njit(cache=True)
def _count_edges_bitboard(h: np.ndarray, v: np.ndarray) -> int:
    """NumPy 配列化された辺の数を数える"""

    # ``np.sum`` で配列全体の合計を求めると Python ループより高速になる
    return int(h.sum() + v.sum())


@njit(cache=True)
def _curve_ratio_bitboard(h: np.ndarray, v: np.ndarray, rows: int, cols: int) -> float:
    """曲率比率を NumPy 配列で高速計算する"""

    # 各頂点に隣接する水平・垂直辺の本数をベクトル化して計算する
    deg_h = np.zeros((rows + 1, cols + 1), dtype=np.uint8)
    deg_v = np.zeros((rows + 1, cols + 1), dtype=np.uint8)

    deg_h[:, :-1] += h
    deg_h[:, 1:] += h
    deg_v[:-1, :] += v
    deg_v[1:, :] += v

    total_deg = deg_h + deg_v
    curves = (total_deg == 2) & (deg_h == 1) & (deg_v == 1)
    curve_count = int(np.sum(curves))

    total = int(h.sum() + v.sum())
    return curve_count / total if total > 0 else 0.0


def _count_edges(edges: Dict[str, List[List[bool]]]) -> int:
    """True の数を数えてループ長を求める"""

    h, v = _pack_edges(edges)
    return int(_count_edges_bitboard(h, v))


def _calculate_curve_ratio(
    edges: Dict[str, List[List[bool]]], size: PuzzleSize
) -> float:
    """ループ中の曲がり角割合を計算"""

    h, v = _pack_edges(edges)
    return float(_curve_ratio_bitboard(h, v, size.rows, size.cols))


def _split_size(total: int) -> list[int] | None:
    """3 と 4 の組み合わせで ``total`` を表す配列を返す"""
    for a in range(total // 3 + 1):
        for b in range(total // 4 + 1):
            if 3 * a + 4 * b == total:
                return [3] * a + [4] * b
    return None


def _copy_pattern(
    dest: Dict[str, List[List[bool]]],
    pattern: Dict[str, List[List[bool]]],
    r_off: int,
    c_off: int,
) -> None:
    """``dest`` の ``(r_off, c_off)`` 位置へ ``pattern`` を貼り付ける"""
    for r, row in enumerate(pattern["horizontal"]):
        for c, val in enumerate(row):
            dest["horizontal"][r_off + r][c_off + c] = val
    for r, row in enumerate(pattern["vertical"]):
        for c, val in enumerate(row):
            dest["vertical"][r_off + r][c_off + c] = val


def _remove_edge_if_exists(
    edges: Dict[str, List[List[bool]]], orientation: str, r: int, c: int
) -> None:
    """存在する場合のみ辺を削除する簡易ヘルパー"""
    arr = edges[orientation]
    if 0 <= r < len(arr) and 0 <= c < len(arr[0]):
        arr[r][c] = False


def _add_path(
    edges: Dict[str, List[List[bool]]], start: tuple[int, int], end: tuple[int, int]
) -> None:
    """開始点から終了点へ直線経路を追加する"""
    r, c = start
    r2, c2 = end
    # 横方向に移動
    while c != c2:
        if c < c2:
            edges["horizontal"][r][c] = True
            c += 1
        else:
            c -= 1
            edges["horizontal"][r][c] = True
    # 縦方向に移動
    while r != r2:
        if r < r2:
            edges["vertical"][r][c] = True
            r += 1
        else:
            r -= 1
            edges["vertical"][r][c] = True


def _connect_tiles(
    edges: Dict[str, List[List[bool]]],
    tile_a: tuple[int, int, int],
    tile_b: tuple[int, int, int],
) -> None:
    """2 つのタイルをパスで接続する"""
    size_a, ra, ca = tile_a
    size_b, rb, cb = tile_b
    start = (ra + size_a, ca + size_a)
    end = (rb, cb)
    # 各タイルの一部を開けておく
    _remove_edge_if_exists(edges, "vertical", ra + size_a - 1, ca + size_a)
    _remove_edge_if_exists(edges, "horizontal", rb, cb)
    _add_path(edges, start, end)


def combine_patterns(
    size: PuzzleSize, rng: random.Random
) -> Dict[str, List[List[bool]]]:
    """登録済みパターンで盤面全体を埋めて一つのループを作る"""

    rows_split = _split_size(size.rows)
    cols_split = _split_size(size.cols)
    if rows_split is None or cols_split is None:
        # 分割できない場合は単純な外周ループを返す
        rows_split = [size.rows]
        cols_split = [size.cols]

    edges = _create_empty_edges(size)
    r_off = 0
    for rs in rows_split:
        c_off = 0
        for cs in cols_split:
            pattern = PATTERNS.get(rs, PATTERNS[3])
            _copy_pattern(edges, pattern, r_off, c_off)
            c_off += cs
        r_off += rs

    # 内部境界を消して外周だけ残す
    for r in range(1, size.rows):
        for c in range(size.cols):
            edges["horizontal"][r][c] = False
    for r in range(size.rows):
        for c in range(1, size.cols):
            edges["vertical"][r][c] = False

    return edges


__all__ = [
    "_create_empty_edges",
    "_generate_random_loop",
    "_apply_rotational_symmetry",
    "_apply_vertical_symmetry",
    "_apply_horizontal_symmetry",
    "_count_edges",
    "_calculate_curve_ratio",
    "combine_patterns",
]


def _warmup_numba() -> None:
    """Numba コンパイルを事前に行うウォームアップ関数"""

    # 1x1 のダミー配列を使い JIT を走らせる
    dummy: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
    _count_edges_bitboard(dummy, dummy)
    _curve_ratio_bitboard(dummy, dummy, 0, 0)


_warmup_numba()
