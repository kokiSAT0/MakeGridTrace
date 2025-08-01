# マップ作成手順

このリポジトリではスリザーリンク用のマップ（問題データ）を自動生成する Python スクリプトを提供しています。ここではスクリプトの実行方法を中心に手順を簡潔にまとめます。関数の詳細な仕様は各ファイルのコメントを参照してください。

## 1. 事前準備

1. **Python 3.12** 以上がインストールされていることを確認してください。Python はプログラミング言語の一種で、コードを書くとさまざまな処理を自動化できます。
2. （任意ですが推奨）仮想環境を作成し有効化します。仮想環境は Python 用の独立した作業領域で、他のプロジェクトとライブラリが混ざらないようにする仕組みです。
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   - `python3 -m venv .venv` で `.venv` という名前の仮想環境を作成します。
   - `source .venv/bin/activate` で仮想環境を有効化します。有効化中は `.venv` ディレクトリにインストールされたライブラリが優先的に使われます。
3. 依存パッケージのインストール
   ```bash
   pip install -r requirements.txt
   ```
   - `pip` は Python の追加ライブラリをインストールするコマンドです。
   - `requirements.txt` には必要なライブラリ名が列挙されています。

## 2. 使い方

### Python から呼び出す

`src/generator.py` の `generate_puzzle` 関数でマップを作成します。必要な引数は行数 `rows`、列数 `cols`、難易度 `difficulty` です。

```python
from src import generator

puzzle = generator.generate_puzzle(4, 4, difficulty="easy")
```

- `rows` と `cols` は盤面サイズ。
- `difficulty` は `"easy"` / `"normal"` / `"hard"` / `"expert"` から選択。
- `timeout_s` を指定するとその秒数で生成処理を打ち切ります。

生成結果は Python の辞書型（`dict`）として得られます。

```python
# 回転対称な盤面を生成したい場合は symmetry="rotational" を指定します
puzzle_sym = generator.generate_puzzle(
    4, 4, difficulty="easy", symmetry="rotational"
)
```

`symmetry` には `"rotational"` (180 度回転) に加えて `"vertical"` (上下反転)
と `"horizontal"` (左右反転) を指定できます。指定した軸で対称となるよう
`solutionEdges` を補正します。

### スクリプトとして実行する

`generator.py` は直接実行することもできます。実行時は少なくとも盤面の行数と列数を指定してください。

```bash
python -m src.generator 4 4 --difficulty normal
```

上記コマンドでは 4×4 の盤面を難易度 `normal` で生成し、`data/map_gridtrace.json` に保存します。保存後は盤面を ASCII 形式で標準出力に表示します。
プロファイルを取りたい場合は `--profile` を付けて実行します。
生成後に `profile.prof` が作成され、`snakeviz profile.prof` で詳細を確認できます。

#### コマンドライン引数一覧

- `rows` : 盤面の行数。必須の数値です。
- `cols` : 盤面の列数。必須の数値です。
- `--difficulty` : 難易度ラベル。`easy` / `normal` / `hard` / `expert` から選択。省略すると `easy`。
- `--symmetry` : `rotational`, `vertical`, `horizontal` のいずれかを指定すると
  回転対称・上下対称・左右対称の盤面を生成します。
- `--theme` : `border`, `maze`, `spiral` のほか `pattern` を指定できます。小さな
  ループパターンを敷き詰めた形状になります。
- `--seed` : 乱数シード。再現したいときに数値を指定します。
- `--timeout` : 生成処理のタイムアウト秒数。指定しない場合は無制限。
- `--parallel` : 並列ワーカー数を指定します。省略すると利用可能な CPU
  コア数だけ起動します。連続して複数回生成する場合はプールを使い回すことで
  初期化コストを抑えられます。
- `--profile` : `cProfile` で計測し結果を `profile.prof` に保存します。`snakeviz profile.prof` で
  グラフィカルに確認できます。
- `--jobs` : `bulk_generator.py` 用の並列生成プロセス数。値を増やすと早く生成できます。

### 繰り返し生成を高速化する

Phase 2.5 から `generate_puzzle_parallel` は軽量なプロセスプールを使う実装に変更されました。
Python から複数回呼び出す場合は次のようにプールを使い回すと効率的です。

```python
from src import generator_parallel

for _ in range(10):
    puzzle = generator_parallel.generate_puzzle_parallel(6, 6, jobs=4)
    # ここで puzzle を利用する
generator_parallel.close_pool()
```
プールを閉じるとワーカープロセスが終了します。

### マルチコア CPU を最大限活用する

特にオプションを指定しない場合、`generate_puzzle_parallel` は
利用可能な CPU コア数だけワーカーを立ち上げます。
すべてのコアを使い切りたいときは `jobs` を指定せずに呼び出します。

```python
from src import generator_parallel

puzzle = generator_parallel.generate_puzzle_parallel(6, 6)
```

コマンドラインから実行する場合も `--parallel` を省略すると
自動的に CPU コア数に合わせてワーカーを起動します。


### 複数の難易度をまとめて生成する

`bulk_generator.py` を使うと、4 種類の難易度を同数生成して一つの JSON ファイルに保存できます。


```bash
# 相対インポートを使っているため `-m` オプションでモジュールとして実行する
python -m src.bulk_generator 4 4 2 --jobs 4
```

1 番目と 2 番目の引数は盤面の行数と列数、3 番目の引数は各難易度で何問生成するかを指定します。`--jobs` を増やすと複数プロセスで並列に生成できます。出力は `data/map_gridtrace.json` に保存されます。

## 3. マップを保存する

1. `save_puzzle` 関数を使うと、生成したマップを JSON ファイルとして保存できます。
   - **JSON**（ジェイソン）はデータをテキスト形式で表現する規格です。多くのプログラムやサービスが対応しています。
   ```python
   path = generator.save_puzzle(puzzle)
   print(f"{path} を作成しました")
   ```
2. デフォルトでは `data/` ディレクトリに `map_gridtrace.json` という名前で保存されます。

## 4. JSON フォーマットについて

保存される JSON の詳細な構造は `slitherlink_map_spec_v1.md` に記載しています。主要な項目だけ簡単にまとめると次のとおりです。

- `id` : マップを一意に識別する文字列
- `size` : 盤面の行数と列数を表すオブジェクト
- `clues` : 出題用ヒント。空白マスは `null`
- `cluesFull` : すべてのセルに数値が入った解答用ヒント
- `solutionEdges` : 正解ループの情報。水平線と垂直線の配列をまとめたもの
- `difficulty` : 難易度ラベル
- `generationParams` : 生成に使った引数を記録したオブジェクト
- `seedHash` : シード値をハッシュ化した文字列

より詳しく知りたい場合は `slitherlink_map_spec_v1.md` を参照してください。

### ヒント数の目安

難易度ごとにヒント密度の範囲を定めています。セル数に `MIN_HINT_RATIO`
と `MAX_HINT_RATIO` を掛けた値が下限・上限となります。

- easy: 28〜38%
- normal: 20〜30%
- hard: 14〜22%
- expert: 10〜18%

生成時にはこの範囲を超えないよう自動調整されます。

## 5. テストの実行

コードを変更したときは `pytest` コマンドでテストを実行できます。テストとは、プログラムが正しく動くか確認する仕組みです。
```bash
pytest
```
時間の掛かるテストを除外して素早く確認したい場合は、以下のように `-m "not slow"` を付けます。
```bash
pytest -m "not slow"
```

## 6. 参考資料

- 盤面生成の仕様書: `MakeGridTraceSPECnew.md`
- JSON フォーマット仕様: `slitherlink_map_spec_v1.md`

以上で基本的なマップ作成の流れは完了です。分からない用語が出てきたら、コメントや仕様書を参照しながら少しずつ理解を深めてみてください。
