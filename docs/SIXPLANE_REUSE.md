# CSPを除く6対象の比較実験

対象は大動脈、腎臓長軸、腸、胆嚢、腹壁と臍帯挿入部、頭蓋外周内領域です。既定runは `sixplane_20261104`、seedは17です。CSPはcohort、分類、共有モデルの出力、条件入力、集計から除きます。旧7対象runは変更しません。

## 準備と再利用の確認

`uv run --locked --no-sync plane-seg prepare-sixplane` は読み取りのみの計画です。`--execute`を付けると、既存7対象runから固定cohortの6対象分を取り出し、6 expertの選択checkpointを変更せずにコピーします。出力runまたは初期重み先が既存なら上書きせず拒否します。原cache、元ZIP、splitは読み取り専用で共有し、CSP画像は新runの許可indexに含めません。

各expertの対象、単一出力、seed、固定checkpoint hash、trainとvalidationの画像ID、前処理・学習条件を照合します。流用元は独立した二値モデルなので、CSP用headやCSPとの競合学習は含みません。新runの実装へstrict loadし、train画像を用いた旧実装とのforward一致も記録します。CSPのゲート変更は、残り6 expertの選択bestが元品質条件を満たすことの代わりにはしません。

共通初期重みは旧runの未学習共通初期値を用います。encoderとdecoderは同一で、未学習headの0〜5番だけを取り出して6出力とします。過去の共有モデルや分類器の学習済み重みは初期化に使いません。初期重みの出典と変換、全学習条件、実行コードとlockの版は非公開runに記録します。

## 実行順

READMEのstage1からstage7までを順に呼び出します。stage1は6分類器の新規学習、stage2は検証済みexpertの再利用のみ、stage3〜5は6出力plain、FiLM、CondConv系の新規学習です。6対象の設定は `configs/sixplane_data_definition.json` と `configs/sixplane_protocol.json` にあります。旧7対象設定は互換性と履歴用に保持します。

準備済みの既定runを最後まで逐次実行する場合は次を使えます。

```powershell
uv run --locked --no-sync python tools/run_sixplane_comparison.py
```

このdriverは再試行や自動再開を行いません。実行中にコード・設定が変わる、他の学習がある、メモリ不足、部分runや品質未達がある場合は次の学習を始めず停止します。変更を確認してから段階CLIで対応します。GPUジョブは逐次です。

## 評価と比較の範囲

全10 checkpointを固定した後にtestを評価します。oracle、分類器argmaxのhard条件、分類確率によるsoft条件、予測channel選択、推定expertへのroutingを区別します。正解マスクは対象channelだけに入り、他5channelの誤検出も別に記録します。6対象の画像内平均を対象ごとに求め、6対象を等重みでmacro集計します。閾値とcheckpoint選択をtestで調整しません。

これは「既存expertを固定して再利用した比較」です。各expertの学習画像・前処理・初期値は対応していますが、expertと共有モデルの総パラメータ数、全画像数、epoch数、計算予算が同一の実験ではありません。旧7対象結果とはタスク・cohort・出力数が違うため、数値の差をそのまま手法の改善と呼びません。単一seedと患者独立性未確認の制約も残ります。

結果、重み、run_manifest、コード版、log、レポートは `runs/` の非公開run内に置き、Gitへ含めません。旧7対象のPDFを6対象の結果で上書きしません。
