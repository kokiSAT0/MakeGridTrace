# AGENTS.md – AI Coding Guide for **MakeGridTrace** Development

> **目的**: このガイドは ChatGPT などの AI コーディング補助を最大限活用し、仕様書 に沿った実装を効率化する。エンジニアが本ガイドの指示をそのまま AI へ投げれば、コンポーネント雛形や関数サンプルを即座に得られるようにする。

---

## 1. プロジェクト概要

- 大人向けスリザーリンク類似ゲーム開発。
- 生成済みマップデータを利用してスリザーリンクを出題するアプリを作成中。
- 本プロジェクトはこのアプリのためのマップデータを作成する。

---

## 2. 技術スタック

- メイン言語: **Python 3.12**
  - Pythonは読みやすさに定評のあるスクリプト言語です。
- テスト: **pytest**
  - pytest はテストコードを自動実行してくれるフレームワークです。
- コード整形: **black**
  - black を使うとコードの書式を自動で統一できます。
- 静的解析: **flake8** & **mypy**
  - flake8 は文法やスタイルの問題点を指摘してくれます。
  - mypy は型ヒント (type hints) を使ってエラーを早期発見するツールです。型ヒントとは変数や引数の型を明示する仕組みです。
- IDE: **VS Code 推奨**
  - VS Code は無料で利用できる開発用エディタで、拡張機能により上記ツールと連携できます.

---

## 3. リポジトリ構成

```
AGENTS.md                  # AI用ガイド
MakeGridTraceSPEC.md       # 盤面生成の仕様書
slitherlink_map_spec_v1.md # マップデータのフォーマット仕様
requirements.txt           # Python依存パッケージ定義
src/                       # 生成スクリプトを置く予定
tests/                     # テストコード
data/                      # 生成したJSONを保存する場所
```

---

## 4. マップデータ 仕様

- slitherlink_map_spec_v1.md に定義。

---

## 5. Git & コミット指針

- 単一ブランチ運用 (`main`)。
- コミット: `feat: canMove 実装`, `fix: 壁衝突バグ` などシンプル & 日本語 OK。

---

## 6. AI  開発基本方針

- **言語**: 日本語
- **実行環境**: Windows11とlinuxに両対応する。

---

## 7. コード品質ルール

- インデントは4スペース。タブは使用しない。
- 変数名・関数名は `snake_case` を用いる。
- 主要な関数には docstring を書いて、何をするか簡単に説明する。
- コメントは日本語でOK。行頭に`#`を付け、なぜその処理が必要かを書く。
- `black` と `flake8` をCIで実行し、書式と文法を自動チェックする。
- 型ヒントを積極的に使うと `mypy` での検査が容易になる。

---

## 8. ビルド & テスト

開発環境は **Python 3.12** 以上を前提とする。

1. 依存パッケージのインストール
   ```bash
   pip install -r requirements.txt
   ```

---


_Last updated: 2025‑07‑02_
