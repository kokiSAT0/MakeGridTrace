# Phase 2 – Quality & Fine‑Tuning (1–2 週間)

> **目的:** 生成速度を維持しつつ、盤面の多様性と解き味を向上させる。
>
> _ターゲット値:_
>
> - Quality Score **中央値 ≥ 70**
> - ヒント密度エントロピー (行列平均) ≥ 0.75
> - 曲率比率 _mean_ ≥ 0.30

## 1. タスクリスト

| ID     | 期待効果 | タスク                                 | 実装メモ                                                                                                    |
| ------ | -------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **B1** | ★★★★☆    | **インクリメンタル Clue 削減**         | 各ヒントの _edge coverage_ を事前計算 → 依存度が低い順に削る → ソルバ呼び出し回数を 5–10 倍削減。 **実装済** |
| **B2** | ★★★★☆    | **Quality Score ウェイト調整**         | `curveRatio` と `hintEntropy` を 0–50%、`solverSteps` 10%、`row/col balance` 15% など。閾値は経験的に校正。 |
| **B3** | ★★★☆☆    | **テーマ拡張 & パターン注入**          | `theme="figure8"`, `theme="labyrinth"` を追加。特定座標に 222, 33, 0 隣接禁止パターンを確率 5–10% で挿入。  |
| **B4** | ★★★☆☆    | **焼きなましパラメータ最適化**         | 温度初期値・冷却率を _board size_ に合わせスケール。小盤面で過加熱、大盤面で収束不足を解消。                |
| **B5** | ★★☆☆☆    | **NumPy/Numba Bitboard 化 (速度維持)** | 辺状態を uint8 配列にパックし、Quality Score 計算を `numba.njit` に置換。速度低下を防ぎつつスコア拡張。     |

> **実装順序:** B1 → B2 → (速度キープ確認) → B3 → B4 → B5

## 2. 評価用ワンライナー

```bash
python - <<'PY'
from bench.local_eval import eval_set  # 自作: 100 盤面生成→統計
print(eval_set(rows=20, cols=20, n=100))
PY
```

- 出力例

  ```
  {'mean_time': 6.1, 'fail_rate': 3.0, 'qscore_median': 72,
   'entropy_mean': 0.83, 'curve_ratio_mean': 0.34}
  ```

## 3. 完了条件

- 速度: Phase 1 の目標を **維持** (微増可)。
- 品質: 上記ターゲット値全てを達成。
- コード品質: TODO リストと FIXME を解消、`profile` でホットスポットに Python 素ループが残らない。

---

> **先行きメモ (Phase 3)**
>
> - 並列生成 (`multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`)
> - ユーザーフィードバックを取り込む Quality Score 再学習
> - ビジュアルヒント配置エディタ連携
