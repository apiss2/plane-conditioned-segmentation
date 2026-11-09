# 最終モデルと実験結果の保存先

普段使う結果は **`runs/final/`** に集約しています。分類器1本とセグメンテーション9本、合計10本の最良checkpointです。`runs/` と `data/` は引き続きGit対象外です。

## 最終モデルの構成

`runs/final/full/` の各フォルダに `best.pt` と設定・学習履歴・選択結果を置きます。

| フォルダ | 内容 |
|---|---|
| `classifier` | 6断面を分類するモデル1本 |
| `expert_0` ～ `expert_5` | 対象ごとのセグメンテーション6本 |
| `plain` | 条件入力なしの共有セグメンテーション1本 |
| `film` | FiLM条件付き共有セグメンテーション1本 |
| `condconv` | CondConv条件付き共有セグメンテーション1本 |

`full/` という既存のモデル配置名は維持していますが、最終結果側には `resume.pt`、optimizer状態、途中checkpointは置きません。各 `best.pt` は2026年11月5日完了の最新比較で固定された重みとbyte一致です。今回の整理で再学習やcheckpointの再選択はしていません。

## 確認するコマンド

リポジトリ直下で実行します。既定の実験名は `final` です。

```powershell
uv run --locked --no-sync plane-seg status
uv run --locked --no-sync plane-seg plan
uv run --locked --no-sync plane-seg verify
uv run --locked --no-sync plane-seg verify --inference
```

`status` と `plan` は10本の最終モデルを表示します。`verify` は保存済み成果のhashを検証し、`--inference` は分類器・expert6本・共有3本をCPUで読み込み、validation画像で再読込一致を確認します。学習やtestセットの再評価はしません。

`final` は読込専用です。学習stage、初期化、データ準備、継続学習、resumeは拒否します。`stage6-evaluate` と `stage7-report` は既存結果を返すだけです。新しい実験を行う場合は、明示的に別の `--experiment` 名を使って準備してください。

OS環境変数の `PCS_EXPERIMENT`・`PCS_RUNS_DIR` や `--env-file .env` は既定値を上書きします。最新結果を読むときは、新規実験用の `.env` と混ぜないでください。`--experiment` は環境変数の実験名より優先されます。

## 結果・設定・来歴

- `final_evaluation/summary.json`: 保存済み比較結果
- `final_evaluation/REPORT.md`: 当時の評価レポート。学習条件差や品質上の留保も保持
- `FINAL_RESULTS.json`: 10モデルの一覧、元checkpointのhash、成果物のhash、archive上の元run
- `FINAL_MODEL_LOCK.json`: 最終結果パッケージの固定重みと読込用設定のhash
- `study_plan.json`: 読込用設定。継続学習の実行設定を除き、学習不可と明示
- `data_references.json`: 同repoの `data/` にある入力・初期重みへの参照
- `provenance/original_study_plan.json` と `original_model_lock.json`: 元の学習設定と固定記録を無変更で保存
- `provenance/execution_source/`: 当時のソースと依存lockの控え

最終パッケージの通常の確認・推論は `runs/archive/` を読みません。ただし入力配列と初期化metadataは `data/` に依存します。この配置整理は、独立した新規test評価や臨床的有効性を保証するものではありません。

## 過去・途中のrun

元の6世代はすべて **`runs/archive/<元の実験名>/`** に移しました。旧7対象、短期6対象、200 epoch版、暫定比較、300 epoch版、500 epoch継続版の元データを含みます。途中状態・resume・optimizer・詳細ログと継続来歴は削除せず、元ファイルをbyte一致で保持しています。

`runs/archive/ARTIFACTS_SHA256.json` が移動前の全ファイルhashです。`ARCHIVE_LAYOUT.json` は配置変更の目印です。CLIは元の `../../data/...` 参照を元の論理位置から解決するため、旧runの記録や登録hashを書き換えません。

```powershell
uv run --locked --no-sync plane-seg --experiment sixplane_radam_sf_continue500_20261105 status
```

この明示指定はarchive上の元runを読みます。archiveのrunはCLIから実行・上書きしません。過去のファイルに記録された絶対パスや元run名は実行指示ではなく来歴であり、現在の保存場所はarchiveです。
