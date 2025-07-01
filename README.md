# マップ作成手順

このリポジトリでは、スリザーリンク用のマップ（問題データ）を自動生成するための Python スクリプトを提供しています。プログラミング初心者の方向けに、専門用語にも簡単な補足を付けながら手順をまとめます。

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

## 2. マップを生成する

1. `src/generator.py` の `generate_puzzle` 関数を使ってマップを生成します。
   - **関数**とは、何らかの処理をまとめたものです。必要な値（引数）を渡すと結果が返ってきます。
   - 例：行数と列数を指定して 4×4 のマップを作る
     ```python
     from src import generator

     puzzle = generator.generate_puzzle(4, 4, difficulty="easy")
     ```
   - `difficulty` は難易度ラベルで、`"easy"`, `"normal"`, `"hard"`, `"expert"` から選びます。

2. 生成されたマップは Python の辞書型（`dict`）として取得できます。辞書型は「キーと値」をセットで保持するデータ形式です。

## 3. マップを保存する

1. `save_puzzle` 関数を使うと、生成したマップを JSON ファイルとして保存できます。
   - **JSON**（ジェイソン）はデータをテキスト形式で表現する規格です。多くのプログラムやサービスが対応しています。
   ```python
   path = generator.save_puzzle(puzzle)
   print(f"{path} を作成しました")
   ```
2. デフォルトでは `data/` ディレクトリに `sl_<行>x<列>_<difficulty>_<timestamp>.json` の形式で保存されます。

## 4. JSON フォーマットについて

保存される JSON の詳細な構造は `slitherlink_map_spec_v1.md` に記載しています。主要な項目だけ簡単にまとめると次のとおりです。

- `id` : マップを一意に識別する文字列
- `size` : 盤面の行数と列数を表すオブジェクト
- `clues` : 各セルのヒント数字を 2 次元配列で保持（空白マスは `null`）
- `solutionEdges` : 正解ループの情報。水平線と垂直線の配列をまとめたもの
- `difficulty` : 難易度ラベル

より詳しく知りたい場合は `slitherlink_map_spec_v1.md` を参照してください。

## 5. テストの実行

コードを変更したときは `pytest` コマンドでテストを実行できます。テストとは、プログラムが正しく動くか確認する仕組みです。
```bash
pytest
```

## 6. 参考資料

- 盤面生成の仕様書: `MakeGridTraceSPEC.md`
- JSON フォーマット仕様: `slitherlink_map_spec_v1.md`

以上で基本的なマップ作成の流れは完了です。分からない用語が出てきたら、コメントや仕様書を参照しながら少しずつ理解を深めてみてください。
