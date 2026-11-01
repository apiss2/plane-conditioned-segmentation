# テストとcoverage / CC / CRAP

リポジトリ直下の `.venv` に、lockで固定した開発環境を構築します。

```powershell
uv sync --locked --group dev
uv run --locked --no-sync pytest -q
```

書式とimport順序は、lockで固定したruff（`ruff format`、`I` ルールを指定した `ruff check`）とローカルのpre-commitフックで統一します。READMEを参照してください。`tests/test_config.py` は、プロトコルテンプレートにすべてのセクションが明示されていることと、不正なプロトコル設定・実行時設定が拒否されることを検証します。

`tests/conftest.py` はテスト収集前に合成metadataを用意し、リポジトリ内の研究用 `runs`、`data`、`archive`、および移行元の実験フォルダへの読込を禁止します。テストでは一時的な合成ZIP・配列・checkpointを使い、外部システムやデバイスの操作を置き換える場合は明示的にmockを使います。本番の学習やtestセットを使った調整は行いません。手動実行のCUDA smokeスクリプトは別扱いで、pytestのcoverageには含めません。

## 計測

```powershell
uv run --locked --no-sync pytest --cov=src/plane_conditioned_segmentation --cov-branch --cov-report=term --cov-report=json:.local/quality/coverage.json --junitxml=.local/quality/tests.xml -q
Copy-Item .local/quality/.coverage .local/quality/measured.coverage
uv run --locked --no-sync python tools/quality_metrics.py --coverage-json .local/quality/coverage.json --coverage-data .local/quality/measured.coverage --output .local/quality/metrics.json
```

アプリケーションの `src/plane_conditioned_segmentation` 配下にあるすべての `.py` ファイルを対象にします。importされないモジュールや `__main__` も含めます。ファイルの除外リスト、行の除外、追加の `no cover` / `no branch` pragmaはありません。テスト・ツール・環境・第三者ライブラリは、このアプリケーションsourceの母集団には含めません。変更前後は同じ母集団で比較します。製品コードの修正で実行可能な行や分岐が増えた場合は、分母を固定したり隠したりせず、その変化も報告します。

Coverage.pyのsubprocess patchにより、隔離した `-I` のfixture検証を含め、子Pythonプロセスも計測します。行coverageと分岐coverageは別々の比率として扱い、pytest-covの表に表示される合算の割合を行coverageとして扱いません。分岐arcの計測は、各論理演算の被演算子まで検証するMC/DC coverageではありません。

## 関数への対応付けとCRAP

`tools/quality_metrics.py` は、AST上の完全修飾名を持つ `def` / `async def` ノードを、コンパイル済みのコードオブジェクトに対応付けます。クラス内のメソッドと入れ子の関数は、それぞれ別の関数として識別します。命令の位置情報から求めた関数本体の行と、coverageが実行可能と判定した行の共通部分を計測します。入れ子の名前付き関数の本体は、親のコードオブジェクトには含めません。ただし、入れ子の関数定義やdecoratorの実行は親に属します。名前付き関数内の無名の内包表記・lambdaは、Radonの関数単位CCと整合するよう、その関数に関連付けます。モジュール直下のlambdaや、モジュール・クラスの本体には関数スコアを付けませんが、source全体のcoverageには含めます。

import時に `def` やdecoratorの行が実行されただけでは、関数本体が実行された証拠にはなりません。本体の実行を計上するには、コードオブジェクトの `co_firstlineno` に対応する負の行番号を起点としたcoverageのエントリーarcを必要とします。これにより、一行関数の定義を実行しただけで、未使用の関数本体まで実行済みと判定することを防ぎます。decoratorの先頭行、入れ子の関数、一行関数、実行されたasync関数、awaitされていないcoroutineは、専用の合成subprocessテストで検証します。Pythonのaudit callbackは通常traceを抑制するため、隔離したcallbackテストだけで `__cantrace__` を設定し、guard本体の実行を計測します。製品コードの動作は変えません。

CCはassertを含めてRadonの `ComplexityVisitor` で求め、入れ子の名前付き関数は個別に採点します。CRAPには分岐coverageではなく、**関数本体の行coverage**を使います。

```text
CRAP = CC² × (1 − covered_body_lines / executable_body_lines)³ + CC
```

traceされていない関数・未実行の関数はcoverageを0として扱い、CRAPはCC²+CCになります。無名コードやdecoratorの先頭行が同じで対応付けが曖昧な場合は、その旨を示し、保守的にcoverageを0として採点します。完全修飾名が重複する定義は、黙って統合せず計測を失敗させます。AST定義を持たない生成関数・native関数は、関数スコアの対象外です。セミコロンで複数の式を一行に詰めたコードでは、coverageだけですべての式の実行を証明することはできません。監査用に、関数に属する行・実行済み行・未実行行の一覧、CC、traceのエントリーの証拠、曖昧な対応付けをレポートに残します。CRAP >30は報告上の閾値であり、不具合の証明でもモデル品質のgateでもありません。

## 保存済みbaseline

変更前後のレポートと生のcoverageデータベースは、`.local/quality` 配下にローカル限定で保存します。baselineのsource・テスト・lockのsnapshotも同じ場所に保持します。sourceを編集した後、元の計測結果からCC/CRAPを再現するには、次を実行します。

```powershell
uv run --locked --no-sync python tools/quality_metrics.py --coverage-json .local/quality/before_coverage.json --coverage-data .local/quality/before.coverage --source-snapshot .local/quality/before_snapshot --output .local/quality/before_reproduced_metrics.json
```

snapshotを使わずに、古いbaselineのcoverage JSONを変更後のsourceへ適用して再計算しないでください。新しくcloneした環境には、これらの非公開の計測成果物は含まれません。合成データやmockを使うテストでは、GPUでの全epoch学習、本番評価全体の制御、benchmark、可視化、外部OS・デバイスとの連携を実証できません。より広い保証を述べる前に、未実行行とリスクのレポートを確認してください。

## 振る舞い契約と反復再現性

入出力・状態遷移の期待値の根拠と、隔離の範囲は [BEHAVIOR_CASES.md](BEHAVIOR_CASES.md) を参照してください。RNGとnative threadの設定は各テストで固定・復元し、呼出元から継承したPCSの入力先設定はクリアし、ZIPのtimestampは固定します。clockを注入するテストにより、再実行で登録内容が書き換わらないことを検証します。

リポジトリ直下で実行します。

    uv run --locked --no-sync python tools/run_behavior_repeats.py
    uv run --locked --no-sync python tools/run_bounded_mutations.py

前者は通常順・逆順で実行し、ケースごとの結果と決定的な観測値の両方を比較します。後者は一時的なsourceコピーだけに変異を加え、変異を加えない対照実行の成功を必須とします。レポートとログは.local/behavior_checksに保存し、研究成果は変更しません。対象を絞った意味上の変異検査であり、網羅的なmutation scoreではありません。Pythonのaudit guardはテストの安全確認用であり、悪意あるnative codeに対するセキュリティsandboxではありません。
