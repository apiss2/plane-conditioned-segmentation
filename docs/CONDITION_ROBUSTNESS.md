# 条件の曖昧さと操作負担を調べる実験

この比較は、既存testの分析を受けて行う探索的実験である。既存testを再評価・選択に使わず、最終的な有用性は新しい患者・施設データと人による操作評価で判断する。

## 実行順序

1. 固定設定のhash確認、空行ラベル、archiveからの継続準備を修正し、回帰テストを通す。
2. `final` のvalidation全256枚についてplain、FiLM、CondConvの条件感度を保存する。
3. FiLMのone-hot、固定smoothing、条件ノイズを、それぞれ新しい初期状態から逐次学習する。
4. 同じvalidation条件で3本を比較し、有望な候補を人による操作評価と複数seed検証へ進める。
5. 未使用の患者・施設データで固定モデル・固定運用を評価する。

## 最初の3条件

| run末尾 | 条件入力 | マスクGT |
|---|---|---|
| onehot | 正解one-hot | 二値のまま |
| smooth | 0.9 × one-hot + 0.1 × 一様分布 | 二値のまま |
| noise | 50%はone-hot。残りは別の5クラスから一つを一様に選び、混合率を0〜0.4から一様に抽出 | 二値のまま |

画像ごと・epochごとに条件を決定するため、worker数・microbatch分割・途中再開で条件は変わらない。
ノイズは合成の対条件ノイズであり、分類器の実際の誤り分布を再現するものではない。

3本ともseed17、同じ未学習初期重み、同じtrain/val、同じ画像拡張と損失、RAdamScheduleFree、最大500 epoch、毎epoch validation、patience30、追加最低予算なし。500は運用上限でありprofiling capではない。checkpoint選択は従来のoracle validation target/global macro Dice平均を維持する。
過去のFiLMには停止条件変更の履歴があるため、新しいonehotを比較対照とする。分類器は固定し、expertは再学習しない。

学習終端のsufficiency判定は保持する。研究用の比較は失敗判定も記録して評価するが、自動採用・閾値緩和はしない。分類器のlabel smoothing、マスク境界のsoft化、条件欠落、OOF分類器の条件、複数seedは最初の3本には混ぜない。

## 条件感度と評価

validationの全画像でoracle、分類器hard/soft、一様混合0.05/0.1/0.2/0.4/1.0、および全5対クラスへの混合0.1/0.2/0.4/1.0を評価する。plainは条件に依存しないため一度だけ測る。

対象別・全体のDice、下位10%、Dice<0.2件数、空マスク件数、oracleから0.1超低下する件数、予測面積の変化を保存する。完全に誤った条件はstress試験として区別し、「誤指定に従わないこと」が常に正しいとはしない。

探索段階の候補条件は、onehot比でoracle macro Diceの低下が0.01以内、分類器softでmacro Diceが非悪化かつDice<0.2件数が減ることとする。これは運用採用基準でも有意差の証明でもない。対象別の悪化、条件を無視していないか、pairedな画像別差も確認する。

## 人による操作評価

独立した評価者が必要。自動推定断面を補助情報として使う場合と、人が抽出対象を明示する場合を別課題にする。

- 対象ごとに成功例、曖昧な条件、大失敗、確認済み陰性を含む症例を用意する。未注釈を陰性と数えない。
- モデル名を隠し、提示順を入れ替える。同一画像を繰り返す場合は順序・慣れの影響を記録する。
- 「そのまま受入」「対象選択の修正」「輪郭修正」「棄却」、受入までの秒数、クリック数、修正量を記録する。
- 確認を求める頻度と、確認を省いた画像に残る失敗率を併記する。平均Diceだけで選ばない。
- 前処理、分類、抽出、表示を含む応答時間を実機で測る。動画はちらつきと安定化による遅延も評価する。

実施者と未使用データの保存先が決まるまでは、操作評価・独立testは未実施として扱う。患者IDを確認し、患者単位で独立性を保つ。施設・装置・画質ごとの症例数、注釈者間差、判断保留例も記録する。

## 実行と停止

```powershell
$env:PCS_EXPERIMENT = 'final'
uv run --locked --no-sync python -m plane_conditioned_segmentation.evaluation.sensitivity --output .local/condition_noise_20261109/baseline
uv run --locked --no-sync python tools/run_condition_ablation.py --output .local/condition_noise_20261109 --prepare
uv run --locked --no-sync python tools/run_condition_ablation.py --output .local/condition_noise_20261109 --execute
```

出力先・run名は新規にする。既存成果を上書きせず、途中失敗は自動再試行しない。各stageの前後で実行ソースとrun設定のhashを検証し、変更されていたら停止する。実行中は対象ソースを編集しない。

現在の学習をbatch境界で止める場合、そのrun直下に `STOP_REQUESTED.json` を作る。比較全体で次のstageを始めない場合は診断フォルダ直下にも同名ファイルを作る。GPUプロセスを強制終了せず、停止後に保存済み状態を確認する。

状態は診断フォルダの `status.json`、各学習の `*_train.log`、runの `full/film/history.json`、比較結果は `comparison.json` に保存する。`screening_complete` は人による評価・独立testの完了を意味しない。
