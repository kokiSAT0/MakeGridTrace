# MakeGridTrace 仕様統合ドキュメント (v2 ベース)

作成日: 2025-07-02
作成者: ChatGPT

---

## 1. 目的

本ドキュメントは `MakeGridTraceSPEC.md` (v1) と `MakeGridTraceSPEC_v2.md` (v2 draft) を統合し、現状の実装状況も踏まえて今後の開発方針をまとめた最新版です。Python でスリザーリンクの盤面データを大量に自動生成する際のガイドラインとして利用します。

旧仕様は `archive/` ディレクトリに移動し参照用とします。以降は本ファイルを参照し開発を進めてください。

---

## 2. 用語集

| 用語            | 説明                                                         |
|-----------------|--------------------------------------------------------------|
| **セル (cell)** | マス目。数字 0~3 または空白 (`null`) を持ちます。           |
| **点 (dot)**    | 格子点 `(r, c)` を表す座標。`r` は行、`c` は列です。        |
| **辺 (edge)**   | 点と点を結ぶ線分。水平線・垂直線の 2 種類があります。       |
| **ループ**      | 分岐や交差が無い 1 本の閉じた線の集合。                     |
| **解の一意性**  | 論理的に導ける正解が 1 通りだけであること。                 |
| **曲率比率**    | ループの総辺数に対する曲がり角 (90°) の割合。               |
| **Quality Score (QS)** | v2 で導入する総合品質指数。値が高いほど良質と判定。 |

各単語はテストコードや実装コメントでも登場します。意味を忘れたらここを参照してください。

---

## 3. API インターフェース

`src/generator.py` で提供される主な関数です。基本的な呼び出し方は次の通りです。

```python
Puzzle = dict  # 生成された盤面データを Python の辞書として扱う型名

from src import generator

puzzle = generator.generate_puzzle(
    rows=5,
    cols=5,
    difficulty="normal",
    seed=1234,
    symmetry=None,
)
```

### 3.1 generate_puzzle

```python
def generate_puzzle(
    rows: int,
    cols: int,
    *,
    difficulty: str = "normal",
    theme: str | None = None,
    symmetry: str | None = None,
    timeout_s: float = 10.0,
    seed: int | None = None,
    return_stats: bool = False,
) -> Puzzle | tuple[Puzzle, dict]:
    """盤面を生成して返します。"""
```

- `rows`, `cols`: 盤面サイズ。4〜30 程度を想定します。
- `difficulty`: `"easy"` / `"normal"` / `"hard"` / `"expert"` の 4 段階。
- `theme`: 盤面の形を決めるオプション。例: `"spiral"`, `"maze"` など。
- `symmetry`: `"rotational"` (180 度回転対称) など対称性を指定できます。
- `timeout_s`: 生成処理のタイムアウト秒数。超過時は品質を妥協して終了します。
- `seed`: 乱数シード。指定すると同じ盤面を再生成できます。
- `return_stats`: `True` にすると生成過程の統計情報も返します。

現行実装では `theme` に `"border"` を指定可能で、`timeout_s` も利用できます。

### 3.2 その他の関数

- `save_puzzle(puzzle, directory="data")`
- `generate_multiple_puzzles(rows, cols, count_each, seed=None)`
- `save_puzzles(puzzles, directory="data")`
- `validate_puzzle(puzzle)`
- `puzzle_to_ascii(puzzle)` : 盤面をテキスト表示する補助関数

---

## 4. 出力 JSON 仕様 (v2 ベース)

生成結果は次のような構造の JSON として保存／返却されます。

| フィールド       | 型      | 説明                                           |
|------------------|---------|------------------------------------------------|
| `schemaVersion`  | string  | `"2.0"` 固定。フォーマットのバージョン管理用。|
| `id`             | string  | 一意な問題 ID。例: `"sl_5x5_easy_20250702"`   |
| `size`           | object  | `{ "rows": int, "cols": int }`               |
| `clues`          | int[][] | ヒント数字 (0–3) または `null`                |
| `solutionEdges`  | object  | 辺情報 `{ "horizontal": bool[][], "vertical": bool[][] }` |
| `difficulty`     | string  | ラベル。生成後の解析結果で上書きされる場合あり |
| `difficultyEval` | string  | ソルバー統計から算出した実測難易度            |
| `loopStats`      | object  | `{ "length": int, "curveRatio": float }`    |
| `qualityScore`   | number  | 0〜100 の品質指数                             |
| `createdBy`      | string  | 例: `"auto-gen-v1"`                           |
| `createdAt`      | string  | ISO-8601 日付文字列                            |
| `symmetry`       | string? | 対称性オプション                               |
| `generationParams` | object | 呼び出しパラメータのエコーバック               |
| `seedHash`       | string  | シード値のハッシュ                             |

現状のコードでは `qualityScore` が計算され、`theme` フィールドも利用できます。

---

## 5. ハード制約 (H-1〜H-9)

| 番号 | 内容                                                    | 検証方法                     |
|-----|---------------------------------------------------------|------------------------------|
| H-1 | **単一閉ループ**を形成する                             | BFS/DFS で連結性を確認       |
| H-2 | **分岐・交差・複数ループなし**                         | 各点の次数を 0 または 2 にチェック |
| H-3 | **ヒント整合** — 各数字セルの周囲 4 辺数が数字と一致   | 盤面走査                     |
| H-4 | **解の一意性**                                         | 簡易ソルバーで解数を数える    |
| H-5 | **JSON 構造厳格準拠**                                   | `jsonschema` などで検証      |
| H-6 | **最小ヒント数** — 難易度別に下限を設定                | ヒント総数を確認             |
| H-7 | **ループ長下限** — 盤面外周だけの短いループ禁止        | 辺数 ≥ 2 × (rows + cols)     |
| H-8 | **(廃止)** v2 では 0 の隣接はソフト制約に変更             | -                         |
| H-9 | **線カーブ比率下限** ≥ 15%                              | `_calculate_curve_ratio` を利用 |

`src/generator.py` の `validate_puzzle` 関数では上記制約を簡易的にチェックしています。

---

## 6. ソフト制約 (品質ガイドライン)

- **難易度ごとのヒント密度**: easy 28–38%, normal 20–30%, hard 14–22%, expert 10–18%
- **ヒント分布**: 盤面全体に散らし、3×3 ブロックすべて `null` などの死角を作らない
- **ループ折返し回数**: 難易度ごとに下限を設け、外周をなぞるだけの単調な形を避ける
- **美観**: `symmetry` オプションで回転対称や鏡映対称を選択可能にする
- **多様性**: 直近生成分との類似度を下げる。将来的に Jaccard 距離で管理予定
- **メタデータ**: `createdBy`, `createdAt`, `difficulty` は必ず付与
- **Quality Score**: 曲率比率、ヒント分散度、ソルバーログ等から計算（実装予定）
- **0 の隣接数ペナルティ**: 隣接する 0 の数に応じて Quality Score を減点

ソフト制約は絶対条件ではありませんが、可能な範囲で守ることでプレイヤー体験が向上します。

---

## 7. 生成パイプライン (v2 設計)

```mermaid
flowchart TD
    A[パラメータ受領] --> B[ベースループ生成 (Improved DFS)]
    B --> C[ヒント導出]
    C --> D[ヒント最適化 (Simulated Annealing)]
    D --> E[一意性 & 制約検証]
    E -- NG --> B
    E -- OK --> F[ソルバートレース & QS]
    F --> G[難易度ラベリング]
    G --> H[JSON 組み立て]
    H --> I[保存 / 返却]
```

現実装では D の焼きなまし最適化や F の QS 計算は未対応です。今後の開発項目とします。

---

## 8. 現状の実装状況

- Python 3.12 対応、pytest によるテストスイートあり
- `generate_puzzle` 関数は v2 仕様の一部 (`schemaVersion`, `loopStats`, `symmetry`) を実装済み
- ハード制約 H-1〜H-9 を `validate_puzzle` で検証済み
- `theme` 引数に `"border"` が利用可能で、Quality Score も計算済み
- CLI スクリプト `python src/generator.py` および `python -m src.bulk_generator` で生成可能

---

## 9. 今後の開発方針

1. **テーマ (`theme`) 拡充**
   - 現在は `"border"` のみ実装済み。今後は複数パターンを追加する
2. **品質指標 (Quality Score) 改良**
   - 曲率比率やヒント分散度に加え、より多くの統計情報から算出する
3. **solverStats 出力**
   - ソルバーが使った手筋回数を記録し JSON に含める
4. **generationParams / seedHash**
   - 呼び出し時の引数をそのまま JSON に残し、シード値はハッシュ化して保存
5. **CI 強化**
   - 生成品質ヒストグラムを作成し、QS の P95 ≥ 70 を目標とする
6. **ドキュメント充実**
   - 各関数の日本語 docstring を整備し、初心者でも追いやすくする

これらを順次実装し、テストを追加して品質を担保します。

---

## 10. テスト指針

- `pytest` を実行し、すべてのテストが緑になることを確認
- 必要に応じてテストケースを拡充し、失敗例を再現できるよう `fixtures` を活用
- 開発環境が変わったら `pip install -r requirements.txt` で依存をそろえる

---

以上が統合仕様書となります。旧版 (`MakeGridTraceSPEC.md`, `MakeGridTraceSPEC_v2.md`) は `archive/` フォルダへ移動し、参考資料として残します。
