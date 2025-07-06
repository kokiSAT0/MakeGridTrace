"""generate_puzzle_parallel 用の軽量プール実装"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Dict, Optional

from .generator import generate_puzzle, setup_logging
from .loop_builder import _calculate_curve_ratio, _count_edges
from .puzzle_builder import _build_puzzle_dict
from .solver import PuzzleSize, calculate_clues, count_solutions
import hashlib

# forkserver を使うことで不要なファイルディスクリプタを継承せず、
# プロセス数が多い場合でも安定して動作する
CTX = mp.get_context("forkserver")

# 使い回すプールを保持するグローバル変数
_pool: Optional[mp.pool.Pool] = None


def _assemble_puzzle(
    rows: int, cols: int, kwargs: Dict[str, Any], data: Dict[str, Any]
) -> Dict[str, Any]:
    """ワーカーから受け取った最小情報からパズル辞書を再構成する"""

    size = PuzzleSize(rows=rows, cols=cols)
    edges = data["solutionEdges"]
    clues = data["clues"]

    # cluesFull は solutionEdges から計算し直す
    clues_full = calculate_clues(edges, size)

    step_limit = kwargs.get("solver_step_limit")
    if step_limit is None:
        step_limit = rows * cols * 25

    # ソルバー統計を計算する
    _, stats = count_solutions(
        clues_full, size, limit=2, step_limit=step_limit, return_stats=True
    )

    loop_length = _count_edges(edges)
    curve_ratio = _calculate_curve_ratio(edges, size)

    generation_params = {
        "rows": rows,
        "cols": cols,
        "difficulty": kwargs.get("difficulty", "normal"),
        "seed": kwargs.get("seed"),
        "symmetry": kwargs.get("symmetry"),
        "theme": kwargs.get("theme"),
        "solverStepLimit": step_limit,
    }

    seed_hash = hashlib.sha256(str(kwargs.get("seed")).encode("utf-8")).hexdigest()

    puzzle = _build_puzzle_dict(
        size=size,
        edges=edges,
        clues=clues,
        clues_full=clues_full,
        loop_length=loop_length,
        curve_ratio=curve_ratio,
        difficulty=generation_params["difficulty"],
        solver_stats=stats,
        symmetry=generation_params["symmetry"],
        theme=generation_params["theme"],
        generation_params=generation_params,
        seed_hash=seed_hash,
        partial=data.get("partial", False),
        reason=data.get("reason"),
    )

    # Quality Score はワーカー計算値を利用
    puzzle["qualityScore"] = data["qscore"]

    return puzzle


def _worker(args: tuple[int, int, Dict[str, Any]]) -> Dict[str, Any]:
    """ワーカー側でパズル生成を実行する関数

    例外が発生しても親プロセスがハングしないよう結果に含めて返す。
    """
    rows, cols, kwargs = args
    try:
        puzzle = generate_puzzle(rows, cols, **kwargs)
        # 軽量化のため必要最小限の情報だけを返す
        trimmed = {
            "clues": puzzle["clues"],
            "solutionEdges": puzzle["solutionEdges"],
            "qscore": puzzle["qualityScore"],
            "partial": puzzle.get("partial", False),
            "reason": puzzle.get("reason"),
        }
        return {"ok": True, "data": trimmed}
    except Exception as exc:  # noqa: BLE001
        # 例外内容を文字列化して親に返す
        return {"ok": False, "err": str(exc)}


def _ensure_pool(jobs: Optional[int], log_level: int) -> mp.pool.Pool:
    """プールを生成し必要なら既存のものを再利用する"""
    global _pool
    if _pool is None:
        proc = jobs if jobs is not None else min(4, CTX.cpu_count())
        _pool = CTX.Pool(
            processes=proc,
            maxtasksperchild=20,
            initializer=setup_logging,
            initargs=(log_level,),
        )
    return _pool


def close_pool() -> None:
    """生成済みプールを終了させるヘルパー"""
    global _pool
    if _pool is not None:
        _pool.terminate()
        _pool.join()
        _pool = None


def generate_puzzle_parallel(rows: int, cols: int, **kwargs: Any) -> Any:
    """並列で ``generate_puzzle`` を実行し結果を返す"""
    jobs = kwargs.pop("jobs", None)
    log_level = kwargs.pop("worker_log_level", logging.WARNING)
    pool = _ensure_pool(jobs, log_level)

    async_res = pool.apply_async(_worker, ((rows, cols, kwargs),))
    timeout = kwargs.get("timeout_s", rows * cols * 2)
    try:
        res = async_res.get(timeout=timeout)
    except CTX.TimeoutError:
        return {"partial": True, "reason": "timeout"}

    if res.get("ok"):
        return _assemble_puzzle(rows, cols, kwargs, res["data"])
    raise RuntimeError("worker failed: " + str(res.get("err")))
