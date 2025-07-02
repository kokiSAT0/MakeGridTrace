
# スリザーリンク マップデータ仕様書 v1.0  
作成日: 2025-07-02  
作成者: Koki

---

## 1. 目的
本ドキュメントは、スリザーリンク（Slitherlink）アプリ用の**問題データ**と**正解ループ**を 1 つの JSON にまとめるフォーマットを定義します。  
マップ制作スタッフは本仕様を満たす JSON を作成してください。アプリ側はこの JSON を読み込み、UI 描画と正解判定を行います。

---

## 2. トップレベル構造

| キー | 型 | 必須 | 説明 |
|------|----|------|------|
| `id` | string | ✔ | 一意な問題 ID。ファイル名や URL に使用可 |
| `size` | object | ✔ | 盤面サイズ `{ "rows": <int>, "cols": <int> }` |
| `clues` | array[][] | ✔ | 出題用ヒント (0–3) または `null`<br>サイズ = `rows × cols` |
| `cluesFull` | array[][] | ✔ | 全てのセルに数値が入った解答用ヒント | 
| `solutionEdges` | object | ✔ | 正解ループの辺情報（後述） |
| `difficulty` | string | 推奨 | `"easy"`, `"normal"`, `"hard"`, `"expert"` など |
| `createdBy` | string | 任意 | 作成者名 |
| `createdAt` | string (ISO-8601) | 任意 | 作成日 `"2025-07-02"` |

---

## 3. `clues` フィールド
- 出題用のヒント数字を 2 次元配列で保持します
- **空白マス**は `null`
- 行と列は **0 インデックス**（配列の行順・列順がそのまま盤面に対応）

### `cluesFull` との違い
`cluesFull` にはすべてのセルに数値が入った解答用ヒントを記録します。`clues` はそ
れを削減した出題データで、`null` のセルがあってもかまいません。アプリは `clues`
を表示し、内部的には `cluesFull` を使って正解判定を行います。

```jsonc
"clues": [
  [null, 3, null, 0],
  [2,    null, 2, null],
  [null, 1, null, 2],
  [0,    null, null, null]
]
```

---

## 4. `solutionEdges` フィールド
盤面の格子点を `(r, c)` = `(0‥rows, 0‥cols)` で表し、  
- `horizontal[r][c]` … 点 `(r,c)` と `(r, c+1)` を結ぶ水平線  
  - サイズ **(rows + 1) × cols**
- `vertical[r][c]` … 点 `(r,c)` と `(r+1, c)` を結ぶ垂直線  
  - サイズ **rows × (cols + 1)**

値は `true` = 線あり / `false` = 線なし。  

#### 例 (4×4 盤)
```jsonc
"solutionEdges": {
  "horizontal": [
    [true,  true,  true,  false],  // r=0
    [false, false, true,  false],  // r=1
    [true,  false, false, false],  // r=2
    [true,  true,  true,  false],  // r=3
    [false, false, false, false]   // r=4
  ],
  "vertical": [
    [true,  false, false, true,  true ], // r=0
    [true,  false, true,  false, false], // r=1
    [false, true,  false, false, true ], // r=2
    [false, false, true,  false, true ]  // r=3
  ]
}
```

---

## 5. 完全サンプル JSON

```jsonc
{
  "id": "slitherlink_4x4_easy_001",
  "size": { "rows": 4, "cols": 4 },
  "clues": [
    [null, 3, null, 0],
    [2,    null, 2, null],
    [null, 1, null, 2],
    [0,    null, null, null]
  ],
  "solutionEdges": {
    "horizontal": [
      [true,  true,  true,  false],
      [false, false, true,  false],
      [true,  false, false, false],
      [true,  true,  true,  false],
      [false, false, false, false]
    ],
    "vertical": [
      [true,  false, false, true,  true ],
      [true,  false, true,  false, false],
      [false, true,  false, false, true ],
      [false, false, true,  false, true ]
    ]
  },
  "difficulty": "easy",
  "createdBy": "Koki",
  "createdAt": "2025-07-02"
}
```

---

## 6. 制作ルール & チェックリスト

1. **整合性**  
   - `clues` の行列数が `size` と一致しているか  
   - `solutionEdges.horizontal` / `vertical` の配列サイズが規定通りか  
2. **ループ条件**  
   - 1 本の閉ループである（分岐・自己交差・分断なし）  
   - すべての数字マスがヒント条件を満たす（0–3）  
3. **難易度タグ**  
   - 主観で良いが、後から一括変更しやすいよう統一語彙を使用  
4. **ファイル名**
  - `map_gridtrace.json` に統一する
5. **レビュー手順**  
   1. JSON Linter で構文チェック  
   2. 内部ツールで自動パズル検証（ヒント整合 & ループ検証）  
   3. 2 名以上で目視確認 → マスターリポジトリへプルリク提出  

---

## 7. 拡張フィールド（任意）

| キー | 型 | 用途例 |
|------|----|--------|
| `tags` | string[] | `"season5"`, `"eventHalloween"` などイベント用分類 |
| `hintCells` | array<{row:int, col:int}> | ユーザがタップしてヒントを得られるセル座標 |
| `timeLimit` | int | クリア制限時間 (秒) |

---

## 8. 参考実装スニペット（抜粋・JavaScript）

```js
import puzzleData from './map_gridtrace.json' assert { type: 'json' };

const { size, clues, solutionEdges } = puzzleData;

// UI 描画例 — ヒント数字
for (let r = 0; r < size.rows; r++) {
  for (let c = 0; c < size.cols; c++) {
    const clue = clues[r][c];
    if (clue !== null) drawNumber(r, c, clue);
  }
}

// 正解判定（水平辺のみ例示）
function isSolved(userEdges) {
  return JSON.stringify(userEdges.horizontal) === JSON.stringify(solutionEdges.horizontal) &&
         JSON.stringify(userEdges.vertical)   === JSON.stringify(solutionEdges.vertical);
}
```

---

## 9. 改版履歴

| 版 | 日付 | 変更概要 |
|----|------|----------|
| 1.0 | 2025-07-02 | 初版リリース |

---

*本仕様についての質問・改善案は Slack #slitherlink-data までご連絡ください。*
