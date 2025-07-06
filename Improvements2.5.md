# Slitherlink Phase 2.5 – Parallel Pool Stabilization

> **目的:** 7×7 以上の盤面生成が _Pool ハング_ により失敗する問題を解消し、並列化を“安全に”活かす。
>
> **ターゲット:**
>
> - 15×15 を ≤ 20 s・成功率 ≥ 70％
> - プール終了で 30 s 固定ブロックが発生しない
> - シングルスレッド性能は維持（Phase 2 目標を下回らない）

---

## A. 原因まとめ（現状観測）

| 症状                          | ログ／プロファイル所見    | 推定原因                                                                 |
| ----------------------------- | ------------------------- | ------------------------------------------------------------------------ |
| `posix.read` が 48 s ブロック | `pool.terminate()` 手続き | ワーカ側プロセスが **クラッシュ or タイムアウト** → 親が Pipe 回収に時間 |
| `Pool.terminate()` に 29 s    | プール全 kill + join      | 盤面ごとにプールを作り直している                                         |

---

## B. タスクリスト（Phase 2.5）

| ID     | 優先  | Task                                    | 実装メモ                                                                                                                                          |
| ------ | ----- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P1** | ★★★★☆ | **ワーカー例外ラップ**                  | `_worker()` 内を `try/except` で囲み `{ok:False, err:str(e)}` を返す。<br>→ 親は `get()` 成功 → エラーメッセージ処理、Pipe が閉じずハングしない。 |
| **P2** | ★★★★☆ | **プール使い回し (`with Pool`)**        | `generate_puzzle_parallel()` を **コンテキストマネージャ** に変更し、同一サイズ連続呼び出しでプール再生成しない。                                 |
| **P3** | ★★★☆☆ | **`get(timeout=…)` でタイムアウト検知** | 親側 `async_res.get(timeout=limit)` → `TimeoutError` で結果捨て、プール継続。<br>Timeout 値は `rows*cols*2` 秒程度から調整。                      |
| **P4** | ★★☆☆☆ | **コンテキスト変更 to `forkserver`**    | `mp.get_context('forkserver')` → フォーク後に不要 FD を持たず、パイプ輻輳を減少。                                                                 |
| **P5** | ★★☆☆☆ | **返却オブジェクト軽量化**              | ワーカーは `{'hints', 'solutionEdges', 'qscore'}` のみ送信。大きい `stats` は親で再計算。                                                         |

---

## C. 実装スケッチ（抜粋）

```python
# generator_parallel.py
import multiprocessing as mp
CTX = mp.get_context('forkserver')  # P4

def _worker(args):
    try:
        data = generate_puzzle_single(*args)
        return {'ok': True, 'data': data}
    except Exception as e:
        return {'ok': False, 'err': str(e)}  # P1

_pool = None  # singleton

def generate_puzzle_parallel(rows, cols, **kw):
    global _pool
    if _pool is None:
        _pool = CTX.Pool(processes=min(4, CTX.cpu_count()),
                         maxtasksperchild=20)  # P2

    async_res = _pool.apply_async(_worker, ((rows, cols, kw),))
    try:
        res = async_res.get(timeout=kw.get('timeout_s', rows*cols*2))  # P3
    except CTX.TimeoutError:
        return {'partial': True, 'reason': 'timeout'}

    if res['ok']:
        return res['data']
    raise RuntimeError('worker failed: ' + res['err'])
```

---

## D. 検証手順

1. **15×15 を 10 回** 生成し、平均時間と成功率を取る。
2. `cProfile -s cumtime` で `posix.read` が上位にいないことを確認。
3. 7×7・10×10 で **シングル vs. 並列** 両方を測定し、Phase 2 と比較。

---

## E. 完了条件

- `posix.read` や `Pool.terminate` が 5 s を超えて出現しない。
- 15×15 平均 ≤ 20 s、成功率 ≥ 70％。
- Quality Score 中央値 ≥ 70 を維持。

---

> **備考:** Phase 3 の SAT ボトルネック削減 (C1/C2) に着手する前に、並列プールの安定化で “ハード障害” を潰すことが目的です。
