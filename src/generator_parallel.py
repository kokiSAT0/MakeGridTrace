"""generate_puzzle_parallel 用の軽量プール実装"""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any, Dict, Optional

from .generator import generate_puzzle, setup_logging

# forkserver を使うことで不要なファイルディスクリプタを継承せず、
# プロセス数が多い場合でも安定して動作する
CTX = mp.get_context("forkserver")

# 使い回すプールを保持するグローバル変数
_pool: Optional[mp.pool.Pool] = None


def _worker(args: tuple[int, int, Dict[str, Any]]) -> Dict[str, Any]:
    """ワーカー側でパズル生成を実行する関数

    例外が発生しても親プロセスがハングしないよう結果に含めて返す。
    """
    rows, cols, kwargs = args
    try:
        data = generate_puzzle(rows, cols, **kwargs)
        return {"ok": True, "data": data}
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
        return res["data"]
    raise RuntimeError("worker failed: " + str(res.get("err")))
