# Slitherlink Map Generator – Phase 1 (Local Edition)

> **目標:** 10×10 盤面を **≤1 s**、20×20 を **≤15 s** で生成し、Quality Score ≥ 50・一意解を維持する。
>
> _CI やクラウドは使わず、ローカル開発だけで回せる手順に絞っています。_

---

## 0. ベースライン計測

```bash
python -m timeit -s "from src.generator import generate_puzzle" \
  "generate_puzzle(rows=10, cols=10, difficulty='normal')"
```

- 10×10, 20×20 の平均生成時間をメモしておく（100 回測定）。
- 以後、変更を入れるたびに同じコマンドで計測して比較。

---

## 1. 実装タスク（優先順）

| ID     | 期待効果 | タスク                                | 実装メモ                                                                                        |
| ------ | -------- | ------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **A1** | ★★★★☆    | **Wilson‑UST でループ生成に置き換え** | 新モジュール `generator/loop_wilson.py` を追加し、`_generate_loop_with_symmetry` から呼び出す。 |
| **A2** | ★★★★☆    | **PySAT (Minisat) で一意解チェック**  | 既存 `count_solutions` を `sat_unique.py` で差し替え。2 解目が出たら即打ち切る。                |
| **A3** | ★★★☆☆    | **動的 `solver_step_limit`**          | `rows * cols * 25` に自動設定。固定 500 000 を廃止。                                            |
| **A4** | ★★★☆☆    | **プロファイラ簡易導入**              | `--profile` フラグで `cProfile` を吐くオプション追加。`snakeviz` で閲覧。                       |
| **A5** | ★★☆☆☆    | **タイムアウト理由の埋め込み**        | 途中返却 JSON に `reason: "timeout"` を追加してデバッグしやすく。                               |

> **実装順序:** A1 → A2 → A3 → A4 → A5

---

## 2. Wilson Algorithm  メモ

1. ランダム頂点を root とし、未訪問頂点から _loop‑erased random walk_ で木を拡張。
2. 完成した木で「葉 →root」の経路を取り、それと root を結んで 1 周ループに。
3. `theme` で重み付けを変えられるよう、歩行時コスト関数を引数に取る。
4. `symmetry` は生成後に `_apply_rotational_symmetry` 等で調整。

---

## 3. SAT エンコード要点

- **変数:** 辺ごとに 1 つ（True=ループ）。
- **次数制約:** 各頂点で ∑Edge ∈ {N,E,S,W} = 0 or 2。2‑SAT で可。
- **一意解チェック:**

  1. モデル ① 取得。
  2. ブロッキング節を追加し、競合制限 1000 で再度 `solve()` 。
  3. 2 解目が見つかれば non‑unique。

---

## 4. 速度改善の目安

| 手順           | 10×10 (平均) | 20×20 (平均) |
| -------------- | ------------ | ------------ |
| **Before**     | 5–10 s       | 30–120 s     |
| **A1 適用**    | ≈0.4 s       | ≈8 s         |
| **A2+A3 適用** | ≈0.3 s       | ≈6 s         |
| **A4+A5**      | 影響なし     | 影響なし     |

---

## 5. クオリティ維持の注意点

- Quality Score  計算は現状ロジックを温存し、閾値  50 を下回った盤面は破棄。
- 失敗率が上がった場合は `solver_step_limit` と `timeout_s` を一時的に緩めて原因を確認。

---

### 完了条件

- 10×10 ≤ 1 s、20×20 ≤ 15 s を **手元の PC** で達成。
- Quality Score ≥ 50 かつ一意解が保証されている。

---

## 付録: 便利コマンド

```bash
# Wilson 法だけを試す簡易スクリプト
python - <<'PY'
from generator.loop_wilson import generate_loop
print(generate_loop(10, 10))
PY

# SAT 一意解チェックの単体テスト
python - <<'PY'
from sat_unique import is_unique
# hints, rows, cols を適当な JSON から読んでチェック
PY
```
