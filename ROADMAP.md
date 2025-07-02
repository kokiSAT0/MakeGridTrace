# 未実装項目チェックリストと開発ロードマップ

このドキュメントでは `MakeGridTraceSPECnew.md` と現行ソースコードを比較し、未実装または今後強化すべき項目を整理します。プログラミング初心者でも追いやすいよう、簡潔な解説を添えています。

## 未実装・強化が必要な項目チェックリスト

- [x] **ヒント最適化(Simulated Annealing)**
  - 仕様書の生成パイプライン D で予定されている焼きなまし最適化を実装しました【F:MakeGridTraceSPECnew.md†L144-L160】。
- [x] **テーマ(`theme`)の拡充**
  - `"maze"` テーマを追加し、ランダム生成から曲がりの多いループを選択するロジックを実装しました【F:src/generator.py†L166-L191】。
- [x] **品質指標(Quality Score)の改良**
  - ヒント密度や行列バランスを評価に加え、より細かな指標で算出するよう更新しました【F:src/puzzle_builder.py†L88-L116】。
- [x] **solverStats の詳細化**
  - 手筋別の枝刈り回数 (``ruleVertex`` / ``ruleClue``) を ``solverStats`` に追加しました【F:src/solver.py†L142-L158】【F:src/puzzle_builder.py†L255-L259】。
- [ ] **CI 強化(品質ヒストグラム)**
  - Quality Score の分布を解析し P95≥70 を目標とする仕組みは未導入【F:MakeGridTraceSPECnew.md†L181-L182】。
- [ ] **ドキュメント整備**
  - 初心者向けの日本語 docstring 充実が課題【F:MakeGridTraceSPECnew.md†L182-L183】。

## 開発ロードマップ(案)

1. **Simulated Annealing の実装**
   - `loop_builder` で生成したヒントを最適化するアルゴリズムを `puzzle_builder` に組み込む。
   - 小さな盤面で試験的に動作させ、品質向上を確認後テストを追加する。
2. **テーマ拡充**
   - `generator.generate_puzzle` の `theme` 引数を拡張し、例として `"spiral"` や `"maze"` などを実装する。
   - テーマごとにループ生成ロジックを分離し、テストケースを用意する。
3. **Quality Score 改良**
   - `_calculate_quality_score` に新たな評価軸(例: 直線率、ヒントの偏り)を追加。
   - 既存テストを基に期待値を調整し、CI に組み込む。
4. **solverStats 詳細化**
   - `solver.py` に解析ログを残す仕組みを入れ、手筋別カウントを JSON に保存する。
   - ログ形式を決め、必要に応じてスキーマを拡張する。
5. **CI の品質計測**
   - 生成した多数の盤面から QS ヒストグラムを作成し、自動的に P95 を算出するスクリプトを用意。
   - GitHub Actions などで定期実行し、目標値を下回ったら警告する。
6. **docstring 充実**
   - 主要モジュールの関数すべてに日本語で簡潔な説明を記述。
   - 関数の入出力例を含むと初心者にも理解しやすい。

以上を順次実施することで、仕様書に沿った完全実装へ近づけます。
