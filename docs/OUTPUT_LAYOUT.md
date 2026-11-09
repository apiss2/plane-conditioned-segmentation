# 現在の出力レイアウト

実験結果の既定保存先はリポジトリ直下の `runs/<実験名>/` です。実行時の作業ディレクトリには依存しません。`runs/` 全体を `.gitignore` で除外し、入力データや初期重みは同じリポジトリ内の `data/` へ分けます。

## 最新結果

既定runは `sixplane_radam_sf_continue500_20261105` です。2026年11月5日08:47 UTC完了の比較で、共有3モデルを総上限500 epochまで継続し、分類器とexpert6本を継承した計10本を扱います。実際の終了epochはモデルごとの早期停止で異なります。

- 保存先: `runs/sixplane_radam_sf_continue500_20261105/`
- 最終集計: `final_evaluation/summary.json`
- レポート: `final_evaluation/REPORT.md`
- モデル・学習履歴: `full/`
- 完了記録: `COMPARISON_COMPLETE.json`
- 固定checkpointのhash: `FINAL_MODEL_LOCK.json`
- 入力データ・初期重みの参照: `data_references.json`

最新214ファイルは既にこの場所に揃っていたため、保存先確認時の再移動や重複コピーは行っていません。画像・重み・評価結果はGitへ追加しません。このドキュメントはローカル成果の場所を示すもので、cloneだけで実験を再現できることを意味しません。

最新結果を確認するときは、新規実験用の `.env` を読み込まず、リポジトリ直下で次を実行します。

```powershell
uv run --locked --no-sync plane-seg status
uv run --locked --no-sync plane-seg verify
```

OS環境変数に `PCS_EXPERIMENT` や `PCS_RUNS_DIR` を設定している場合は、そちらが優先されます。`--experiment` は環境変数の実験名より優先されます。入力先の `PCS_DATA_DIR`、`PCS_ARCHIVES_DIR`、`PCS_INITIALIZATION_DIR` も指定されていれば保存済み参照より優先されるため、既存結果と異なるデータを混ぜないでください。新しい実験は必ず別名で準備し、完了結果を上書きしないでください。

## 旧runをarchiveへまとめる条件

`runs/archive/` への移動対象は、実行中の処理・設定・準備コマンド・結果の参照・再現来歴から完全に未使用と確認できるものだけです。単に古いという理由では移動しません。

2026年11月9日の確認では、旧7対象・短期6対象・200 epoch・300 epochのrunは準備元や継続元として参照され、暫定比較runも200 epoch側の評価記録と再利用処理から参照されていました。完全未使用と断定できるrunはなく、移動・削除は行っていません。最新runの通常の読込は、自身のcheckpointと `data/` の入力に依存します。旧runへの来歴参照と、最新runの実行時の依存は区別します。

`archive` へ階層を1段深くして単純移動すると、旧run内の `../../data/...` が指す場所や準備元の検索が変わります。移動する場合は、参照先の検証、全ファイルのhash照合、旧成果を再利用する手順の確認を伴う別の変更として扱ってください。
