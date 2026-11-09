# 6対象のRAdamScheduleFree実行規則

この設定はCSPを除く6対象を扱い、分類器1・expert6・plain1・FiLM1・CondConv1を初期重みから学習する。旧7対象と短期6対象の成果は変更しない。結果と画像・重みは非公開の `runs/` 配下へ保存する。

## 固定する条件

公式 `schedulefree==1.4.1` の `RAdamScheduleFree` を使う。APIは公式ソースと実インストールで照合する。encoder LR `1e-5`、その他LR `1e-3`、weight decay `1e-4`、betas `(0.9,0.999)`、eps `1e-8`、r `0`、weight_lr_power `2`、foreach `True`、silent_sgd_phase `True`。外部schedulerとwarmupはない。初期数stepはRAdamの内部仕様で更新が抑制される。学習率はgroup LRと内部rectified LRの両方をhistoryへ記録する。

LRは従来値を保ち、testを見て選ばない。この値が最適だと主張しない。RAdamScheduleFreeのweight decayはtraining yでのL2項であり、旧AdamWのdecoupled decayと同一ではない。旧optimizer状態は流用しない。

最大200 epoch。毎epoch validationを行い、分類器はmacro recall、segmentationは対象macro Diceと全出力global macro Diceの平均を監視する。`min_delta=0`の厳密な改善でcounterを0へ戻す。同値・低下は1加算し、30回連続で非改善なら停止する。追加の最低60 epoch・1,500更新は課さない。実装のmin_epochs=1は最初のvalidationがepoch1であることを表し、追加予算ではない。NaN/Infは例外で停止し、bestは厳密最大値、同値なら早いepochを残す。

旧train停滞guardはこの経路のearly stoppingとして使用しない。terminal train品質ゲートは旧閾値を維持した別のレビュー基準で、validation選択bestの品質保証とは区別する。閾値未達時に自動で緩めない。

## モードと保存

学習では `optimizer.train()` と `model.train()`、validation・推論・保存では `optimizer.eval()` と `model.eval()` を使う。`best.pt` は評価用の平均化重みxだけを持つ。`resume.pt` は同じxに加えてoptimizerのz/二乗モーメント、mode、step数、scaler、RNG、history、停止counter、設定hashを持つ。再開時は両方を読んでからoptimizer.trainへ切り替える。保存・再開の途中で旧AdamW状態を読み込まない。

encoderのBatchNorm running statisticsは元設計どおり固定、decoderはGroupNormである。新たなBN再推定は行わない。学習モードのyと評価モードのxを行き来するfloat32演算には丸め誤差があり、厳密なビット一致が必要な検査と許容誤差を伴う再開検査を区別する。

`resume-model --model NAME` は状態確認のみ、`--execute`を付けると明示再開する。完了モデル・設定変更・異なるoptimizer・source変更・不整合状態は拒否し、自動上書きや `--force` は提供しない。

## 順序・暫定比較・制約

順序は分類器→plain→FiLM→CondConv→暫定評価→新expert6→最終評価。共有3本が完了した時点で、旧固定expertをコピーした別runに暫定比較を保存する。旧expertはAdamW＋cosineであり、全モデル同じoptimizerの公平な最終比較とは区別する。暫定testを見て後続モデルの条件・checkpoint・閾値を変更しない。最終評価前に新10本を固定し、testは選択に使わない。

seedは17のみ。元splitと隔離済み重複を保つが、患者独立性は未確認。過去実験で既に観察されたholdoutの再利用であり、独立した新しい外部検証ではない。旧低解像度実験との直接的優劣も、GTや評価gridが違う場合は主張しない。

一次資料: [公式README](https://github.com/facebookresearch/schedule_free)、[RAdam実装](https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/radam_schedulefree.py)、[PyPI](https://pypi.org/project/schedulefree/1.4.1/)。

## 既存runの停止条件を訂正するとき

以前のrunには依頼外の最低60 epoch・1,500適用更新が含まれていた。完了済み履歴と重みを上書きせず、基準のstudy_planを来歴として保持する。残りモデルはprotocol hashに結び付けたstopping_policy_override.jsonで最低条件だけを外し、進行中モデルはoptimizer・RNG・counter・部分batchを保存して明示再開する。元configurationとbest/resumeを保存し、effective configurationと切替記録を追加する。
過去の履歴から求めた本来の停止時点と、当時の重みが実際に残るかを区別する。スコアだけからcheckpointを再現したとは扱わない。モデルごとの条件差、余分な学習、選択重みへの影響を非公開runの比較レポートに記載する。test成績を変更理由にしない。


## 200 epoch以降の共有モデル継続

暫定PDF提出後の明示指示に限り、`prepare-continuation` で完了runとは別の兄弟runを作る。plain・FiLM・CondConvのみを終端200の完全resume状態から継承し、総上限300、patience30、追加最低epoch/更新数なしとする。既存best・200 epochの履歴・非改善カウンタ・optimizer・scaler・RNGは維持する。元のresumeをbyte一致の控えとして保持し、新resumeで変更するのはrun設定hashとearly-stopping policyのみ。全数値状態の完全一致を検査する。分類器と6 expertはbyte一致コピーを再利用し、再学習しない。旧runのtest・重み・履歴・PDFを変更しない。

```powershell
uv run --locked --no-sync plane-seg --experiment sixplane_radam_sf_continue_20261105 prepare-continuation
uv run --locked --no-sync plane-seg --experiment sixplane_radam_sf_continue_20261105 prepare-continuation --execute
$env:PCS_EXPERIMENT = "sixplane_radam_sf_continue_20261105"
uv run --locked --no-sync python tools/run_sixplane_comparison.py
```

既存の完了runを直接resumeする保護は維持する。共有stageのpartial状態だけを明示resumeし、完了後は通常stageで再利用する。推論用bestに終端optimizerを混ぜない。検証指標だけで停止・bestを選択し、testは全モデル固定後に評価する。許可済みの上限またはpatience停止で収束診断だけが未達なら、False判定と閾値を保持した注意書き付きで比較を続行できる。低いtrain品質・破損状態・無関係な例外は引き続き停止する。最初の200 epochとexpertの追加最低条件による来歴の留保も消さない。


## 300 epochから500 epochへの継続

追加200 epochは必ず200回実行する意味ではなく、総上限500とする。最新指示によりpatience30の早期停止を適用し、300時点のbest・reference・非改善カウンタを保持する。最低epochや最低更新数を追加しない。optimizer・scaler・RNG・300 epochの履歴は同一状態で継承し、分類器と6 expertは再利用する。元runを直接再開せず別runを作る。

```powershell
uv run --locked --no-sync plane-seg --experiment sixplane_radam_sf_continue500_20261105 prepare-continuation --source-experiment sixplane_radam_sf_continue_20261105 --max-epochs 500 --execute
$env:PCS_EXPERIMENT = "sixplane_radam_sf_continue500_20261105"
uv run --locked --no-sync python tools/run_sixplane_comparison.py
```

選択・停止はvalidationのみで決める。既に観測済みのtestを再利用した比較は探索的評価であり、独立した確認用testではない。test値による条件・閾値調整はしない。古い200/300 epoch結果・重み・PDFを上書きしない。
