# 振る舞い契約：範囲を絞ったテストmatrix

データとcheckpointはすべて合成し、元の研究成果物へのアクセスは禁止します。これらのケースでは、mockの呼出しやcoverageの割合だけでなく、観測できる数値・ファイル・状態遷移を検証します。

| ID | パターン | 期待値の根拠・期待する振る舞い | テスト |
|---|---|---|---|
| M1 | 予測なし、FNが1画素、完全一致、FPが1画素、予測と正解が重ならないFP＋FN | 画素集合を手で列挙してTP/FP/FN/PC/TCの正確な個数を求め、それらの独立した個数からDiceを計算する | `test_hand_counted_fp_fn_dice` |
| M2 | sigmoidが0.5未満・0.5ちょうど・0.5超 | 厳密に `>0.5` と判定し、logitがゼロなら陰性とする | `test_strict_half_probability_threshold` |
| M3 | padding画素内に予測や正解（GT）がある | paddingはFPにもFNにも加算せず、有効領域にある1画素のFNは残る | `test_padding_never_counts_as_fp_or_fn` |
| M4 | 正しい対象と、意味的に誤ったchannelへの予測がある | 対象のDiceは変わらず、global Diceと対象外channelのFPで誤ったchannelへの予測を捉える | `test_offtask_false_positive_does_not_change_target_dice` |
| M5 | GTが空で、予測がある場合・ない場合 | 空集合のepsilon規約を明示し、臨床的に対象が存在しないことの証拠とは扱わない | `test_empty_ground_truth_has_explicit_set_semantics` |
| L1 | 既知のGTが4画素あり、logitがゼロ | 対象の損失の閉形式は `ln2+2/7`、共有モデルで加わる負例channelの項は `ln2` | `test_zero_logit_loss_has_closed_form_oracle` |
| I1 | logit・GT・有効画素maskにNaN/+Inf/-Infがある | NaNを予測陰性や正解ラベルへ黙って変換せず、失敗させる | `test_nonfinite_tensors_are_rejected_not_silently_scored_as_fn` |
| I2 | 有効画素なし、小数・負値のmask、小数・範囲外の断面ID、形状不一致 | 指標や損失を計算する前に明示的に拒否する | `test_invalid_metric_and_loss_contracts` |
| I3 | 成立しないDice・FP率、負の画素数 | 不可能な値を正当な集計値として扱わない | `test_aggregate_rejects_impossible_metrics` |
| G1 | 4つのgateそれぞれについて、閾値の直前・一致・直後 | IEEEの `nextafter` で隣接値を作り、下限とgainの上限はともに境界を含める。閾値や許容誤差は緩和しない | `test_gate_ieee_boundary_contract` |
| G2 | 空履歴、NaN、Inf、epochの重複・逆順、終端指標の欠落、対象集合の変化 | 不正な履歴を拒否し、別の状態を終端として代用したり、基準未達の対象を除外したりしない | `test_bad_history_cannot_pass_or_choose_a_different_terminal` |
| S1 | 計画→初回登録→異なるclockでの再実行→改変 | 計画では書き込まない。初回のclockを明示し、再実行後もbytesが同一であること、改変を拒否して元の登録を保持することを確かめる | `test_registration_plan_register_replay_tamper_state_machine` |
| S2 | lock取得→別の所有者による取得試行→例外→再取得 | 別の所有者は実行区間へ入れず元のlockも削除できない。例外時は自分が所有するlockだけを片付け、再取得は成功する | `test_real_execution_lock_state_transitions_and_exception_cleanup` |
| W1 | 実モデル・分類器の保存→読込→推論の反復 | 実際のtorch直列化とnetworkのforwardを使い、CPU出力の厳密な反復一致とcheckpoint・状態の不変を確かめる | `test_actual_checkpoint_roundtrip_and_eval_repeatability`、分類器側の対応するテスト |
| E1 | 準備済みcache一式と、未学習の新規checkpoint 11個 | 実際のZIP準備・loader・torch load・CPU networkを使い、ゼロを返すだけの偽モデル・偽loaderは使わない | `test_verify_real_cache_and_real_untrained_checkpoint_reload` |
| H1 | ネットワーク・研究用ファイルへの書込event | 実際の接続やファイル作成の前に、audit eventの方針で拒否する | `test_harness_contracts.py` |
| H2 | 継承された不正なPCSパス、実行順やRNGの変化 | ハーネスが呼出元のPCS入力先をクリアし、RNG・thread状態を復元して固定乱数列を使う。通常順・逆順の両方で成功することを必須とする | conftestと反復実行スクリプト |
| D1 | 同じfixture入力から2か所へデータを準備し、同じ出力先にも再実行する | 配列・manifest・geometryのbytesが一致することと、既存出力先を上書きせず拒否することを確かめる | 既存のportabilityテスト |

## 隔離と冪等性の範囲

- 書き込むのは、一時的な入力fixture・cache配列・未学習checkpoint・テスト用runの登録ファイル・Git除外の監査出力だけです。optimizerによる更新、学習epoch、臨床データの評価は実行しません。
- 入力のseedとアーカイブentryのtimestampを固定します。登録テストでは固定clockを注入し、意図的に時刻を進めても再実行で状態が書き換わらないことを確かめます。
- RNG、native thread設定、呼出元のPCS環境を隔離・復元します。ネットワークと元の研究成果物・ローカル研究成果物へのアクセスは禁止します。変異検査で変更するのは一時的なsourceコピーだけです。
- `plan`、`status`、完了stageの再利用、登録の再実行、データ読込は冪等であることを期待します。新規run・cacheの作成は排他的で、2回目は上書きせず拒否します。event logは設計上追記専用であり、冪等とは扱いません。
- 既存のOS/GPU照会をmockするテストは、adapter・guardの検証に限ります。実機の振る舞いを証明するものではありません。座標テストでCPUへ転送を置き換える点も明示します。実モデルの保存・読込往復と再読込の結合テストでは、数値ロジックを偽の予測で置き換えません。
- 残る制約：GPUでの全epochループ・AMP・OOM、性能benchmark、すべてのfilesystem・network実装、platformをまたぐbit単位の一致、考えられるすべての不正metadataや変異は、この範囲を絞った検査では網羅していません。
