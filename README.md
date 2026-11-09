# 断面情報を使う胎児超音波画像の領域抽出

胎児超音波画像から対象の領域を抽出する際に、**「どの断面の画像か」という情報をモデルへ渡すと、抽出結果や誤検出がどう変わるか**を比較する研究用プログラムです。

画像の断面を判定する分類器、断面ごとに分けて学習するモデル、複数断面をまとめて学習するモデルを、同じデータ分割と正解マスクで比較します。実験結果・画像・学習済み重みはGitに含めません。**このリポジトリをcloneしただけでは、データの準備や過去の実験の再現はできません。**

最新結果の保存先と既定runは [現在の出力レイアウト](docs/OUTPUT_LAYOUT.md) を参照してください。以下に記載された旧実験名とは区別します。

## 何を比較するのか

対象は7種類の断面・抽出対象の組です。対応するラベルと元ZIPは [configs/data_definition.json](configs/data_definition.json) に定義しています。

| ID | 抽出対象のラベル |
|---|---|
| 0 | 大動脈（`aorta`） |
| 1 | 腎臓の長軸断面（`kidney_longitudinal`） |
| 2 | 腸管（`intestine`） |
| 3 | 胆嚢（`gallbladder`） |
| 4 | 腹壁・臍帯付着部（`wall_and_cord_insertion`） |
| 5 | 頭蓋輪郭内の領域（`calvaria_enclosed_region`） |
| 6 | 側脳室断面の透明中隔腔（`csp_transventricular`） |

領域抽出には、グレースケール画像を入力するMobileNetV2ベースのU-Netを使います。比較する方式は次の4つです。

| 方式 | 仕組み |
|---|---|
| expert | 断面ごとに専用モデルを1つ作る。各モデルの出力は、その断面の対象領域1つ |
| plain | 1つの共有モデルから7対象のマスクを出力する。断面情報はモデルへ渡さない |
| FiLM | 断面情報に応じて、共有モデルの途中の特徴量を拡大・縮小したり、値を加えたりする |
| CondConv系の方式 | 断面情報に応じて複数の畳み込みkernelを混ぜ、共有モデルの処理を変える |

この実装のkernel混合は、画像から求める条件ではなく、外から与えた断面情報を使います。各方式の出力について、正解領域との重なりを表すDiceと、対象外channelへの誤検出などを比較します。断面情報は、正解の断面が分かっている場合と、独立した分類器で推定する場合を区別して評価します。

学習するモデルは合計11個です。内訳は、断面分類器1個、expert 7個、共有モデルのplain・FiLM・CondConv系が各1個です。

## 実行前に必要なもの

- Python 3.13.2と[uv](https://docs.astral.sh/uv/)：uvは、このプログラムが使うPythonライブラリの導入と実行をまとめて行うツールです。
- 学習に使うNVIDIA GPUと、CUDA版PyTorchに対応するdriver。
- 利用許諾を確認した元画像・YOLO形式のpolygonラベルを含むZIP。
- 各画像のtrain/val/test、使用可否、画像の配置情報を記したmanifest。
- 初期化用の**1channel MobileNetV2 encoderの`state_dict`**と、`gray_mean`・`gray_std`を含む正規化情報のJSON。

元ZIPだけでは実行できません。元データの取得、画像へのラベル付け、患者情報に基づくデータ分割は利用者が準備してください。データや事前学習重みを自動でダウンロードしたり、RGBの重みを黙って変換したりする機能はありません。

## 1. Pythonの実行環境を作る

以下の例はPowerShell用です。最初に、このREADMEがあるフォルダを作業ディレクトリにしてください。

```powershell
uv sync --locked
```

必要なライブラリが`uv.lock`で指定した版で導入されます。uvがリポジトリ直下に作る`.venv`は、このプロジェクト専用のPython実行環境です。実験フォルダごとに別の環境を作る必要はありません。

## 2. 実験名とデータの保存先を決める

```powershell
Copy-Item .env.example .env
```

作成した`.env`を編集します。次の例では、新しい実験を`my_sevenplane_run`という名前で保存します。

```dotenv
PCS_EXPERIMENT=my_sevenplane_run
PCS_RUNS_DIR=./runs
PCS_DATA_DIR=./data/my_sevenplane_run/cache
PCS_ARCHIVES_DIR=./data/my_sevenplane_run/source_archives
PCS_INITIALIZATION_DIR=./data/my_sevenplane_run/initialization
```

| 設定 | 指定する場所・名前 |
|---|---|
| `PCS_EXPERIMENT` | 実験を区別する名前 |
| `PCS_RUNS_DIR` | 重み・設定・評価などの出力先。実験名のフォルダがこの下にできる |
| `PCS_DATA_DIR` | 元画像から作る学習用配列の保存先 |
| `PCS_ARCHIVES_DIR` | 元ZIPを置くフォルダ |
| `PCS_INITIALIZATION_DIR` | 初期化用の重みと正規化情報を保存するフォルダ |

相対パスは実行時の作業ディレクトリが基準です。外部ドライブの保存先も指定できます。空白を含む値は引用符で囲んでください。

`.env`は自動では読み込まれません。以降の例のように`uv run --env-file .env ...`で指定します。OSの環境変数として設定することもできます。`--experiment`を明示した場合は、環境変数の実験名より優先されます。

既存の実験を開くときは、実験名と入力先をその実験に合わせてください。入力先の3変数を省略すると、保存済みの`data_references.json`などから解決します。異なるデータを既存モデルと混ぜないように、新しい実験には別の名前を使ってください。

## 3. 元画像とラベルから入力データを作る

元ZIPを`PCS_ARCHIVES_DIR`のフォルダへ置き、画像ごとのmanifestをJSON配列で用意します。以下は架空の形式例です。

```json
[
  {
    "index": 0, "plane": 1,
    "archive": "abdomcirc_segmentare.zip",
    "name": "sample/images/example.png",
    "label": "sample/labels/example.txt",
    "split": "train", "allowed": true,
    "geometry": {
      "original_width": 1024, "original_height": 768,
      "input_width": 544, "input_height": 384,
      "content_width": 512, "content_height": 384,
      "left": 16, "top": 0
    }
  }
]
```

- `index`は0からの連番、`plane`は元データの断面IDです。抽出対象IDとの対応は`data_definition.json`を参照してください。
- `name`と`label`はZIP内の画像・ラベルのパスです。
- `split`は`train`・`val`・`test`のいずれか、`allowed`は使用可否を表す真偽値です。
- `geometry`は元画像の大きさ、入力画像の大きさ、余白を除く内容の大きさと配置を整数で指定します。入力サイズはプロトコル設定と一致させます。
- 任意の`sha256`は、グレースケール画素bytesと元画像サイズの文字列を連結した内容のSHA256です。

実際のmanifestには、比較する各対象の画像と固定したデータ分割を記載します。プログラムが患者groupや元splitを推測することはありません。splitをまたぐ完全重複画像は拒否しますが、近似重複や患者の独立性まで保証するものではありません。

```powershell
uv run --env-file .env --no-sync plane-seg prepare-data --manifest ./data/private_manifest.json
uv run --env-file .env --no-sync plane-seg check-data
```

`prepare-data`は、プロトコルに従って画像のサイズと余白を調整し、マスクを作って配列へ保存します。同時に、その実験用の設定とデータ参照も作ります。既存の実験・入力配列は上書きしません。`check-data`で保存先・画像数・読み込んだ画像の形状などを確認してください。

## 4. 共通の初期重みを用意する

用意したencoderの重みと正規化情報を指定します。

```powershell
uv run --env-file .env --no-sync plane-seg initialize-models --encoder ./data/encoder.pt --metadata ./data/normalization.json
```

encoderは1channel MobileNetV2の`state_dict`である必要があります。正規化JSONには`gray_mean`と`gray_std`を指定します。この処理で、各方式に使う共通の初期状態を作ります。既存の初期化ファイルは上書きしません。

## 5. 分類器・expert・共有モデルを順に学習する

まず状態と実行計画を確認します。この時点では学習しません。

```powershell
uv run --env-file .env --no-sync plane-seg status
uv run --env-file .env --no-sync plane-seg plan
uv run --env-file .env --no-sync plane-seg stage1-classifier --plan
```

実際に学習するには、各段階に`--execute`を付け、順に実行します。

```powershell
uv run --env-file .env --no-sync plane-seg stage1-classifier --execute
uv run --env-file .env --no-sync plane-seg stage2-experts --execute
uv run --env-file .env --no-sync plane-seg stage3-baseline --execute
uv run --env-file .env --no-sync plane-seg stage4-film --execute
uv run --env-file .env --no-sync plane-seg stage5-condconv --execute
```

expertは`--target 0`のように対象を限定することもできます。各段階は前段階の完了を確認します。既に完了したモデルは再利用し、学習途中の結果・品質基準未達・実行競合がある場合は停止します。自動的な上書きや強制再開はしません。`--force`はありません。GPUジョブは逐次実行します。

## 6. 固定したモデルを評価し、レポートを作る

```powershell
uv run --env-file .env --no-sync plane-seg stage6-evaluate --execute
uv run --env-file .env --no-sync plane-seg stage7-report --execute
```

評価前に全11モデルのcheckpointを固定します。testセットでの評価はその後に行い、test結果を使ってモデルや閾値を選び直すことは禁止します。既存の評価・レポートがあれば再利用します。

既存モデルの読込を追加で確認する場合は、次のコマンドを使います。validation画像で再読込結果を比較する処理であり、test評価や再学習ではありません。

```powershell
uv run --env-file .env --no-sync plane-seg verify --inference
```

既にそろっている完了成果を登録する`register-existing --execute`もあります。これは新たな学習・評価を行わず、保存済み成果の整合性を確認して`run_manifest.json`を作るための処理です。

## 結果はどこに保存されるか

設定例の場合、実行成果は`runs/my_sevenplane_run/`へまとまります。

| 場所 | 内容 |
|---|---|
| `full/<model>/` | 各モデルの重み、学習設定、履歴、結果 |
| `final_evaluation/summary.json` | 比較結果の集計 |
| `final_evaluation/REPORT.md` | 評価レポート |
| `study_plan.json` | その実験で使用する固定設定 |
| `data_references.json` | 入力データ・初期化ファイルの参照先 |
| `FINAL_MODEL_LOCK.json` | 評価に使用するcheckpointと設定の取り違え防止用記録 |
| `run_manifest.json` | 登録した完了成果と整合性情報 |

移行済みの実験には、追加の集計や報告が`aggregates/`・`reports/`に保存されている場合もあります。入力データは`data/<run>/`、使用しない旧コードや移行前の控えは`archive/<run>/`、ローカルの検査記録は`.local/`へ分けます。これらはすべてGit除外で、結果を公開するフォルダは設けていません。

## 条件を変えるときに確認する設定

| ファイル | 役割 |
|---|---|
| [configs/data_definition.json](configs/data_definition.json) | 対象と元データのラベル対応 |
| [configs/sevenplane_protocol.json](configs/sevenplane_protocol.json) | 前処理、データ拡張、学習・最適化、品質基準、評価方法のテンプレート |
| [configs/runtime.json](configs/runtime.json) | CPU thread数、読込worker、評価batch、必要GPUメモリなど実行環境の設定 |

結果に関わる設定は、データ準備時に実験内の`study_plan.json`へ写し、その後はこの固定コピーから読みます。テンプレートを編集しても、既存実験の条件は変わりません。以前に作成した実験で不足するセクションは、`config.py`の`LEGACY_V1_SECTIONS`に記録した当時の値で扱います。

必須設定の不足、設定セクション内の不明な項目、型・seed・入力サイズやbatchの不整合を検証し、問題があれば拒否します。7対象・11モデル・seed 17は現在の研究構成として固定しています。また、encoderやdecoderなどのモデル構造を変えると既存checkpointを読めなくなるため、自由に差し替えられる設定にはしていません。

## 開発・テスト

アプリケーションのコードは`src/plane_conditioned_segmentation/`、テストは`tests/`にあります。開発用ツールも導入する場合は、次を実行します。

```powershell
uv sync --locked --group dev
uv run --locked --no-sync pytest -q
uv run --locked --no-sync ruff check .
uv run --locked --no-sync ruff format --check .
```

commit時の書式・import順検査を使う場合は、ローカルにフックを登録します。

```powershell
uv run --locked --no-sync pre-commit install
```

ruffの版は`uv.lock`、設定は`pyproject.toml`にあります。現在のlintはimport順の`I`ルールのみです。pre-commitはローカルの固定ruffを使い、フック用リポジトリはダウンロードしません。テストの方法と範囲は [docs/TESTING.md](docs/TESTING.md)、入出力・境界・状態遷移の一覧は [docs/BEHAVIOR_CASES.md](docs/BEHAVIOR_CASES.md) を参照してください。

## 制約

- 過去のmanifest・cohort（使用画像の集合）・データ・重みは同梱していません。厳密な再現には、元の非公開資料も必要です。
- 実医療データ全件の再生成一致、患者・施設の独立性、臨床での有用性を保証するものではありません。
- 選択したcheckpointの品質と、学習終端の十分性は別に扱います。基準未達や評価方針変更の留保は、各実験の記録に残します。
- 元データと第三者重みの利用・再配布許諾、プロジェクトのLICENSEは利用者が確認する必要があります。
- データを任意のGit対象フォルダへ置くと、ignoreだけでは保護できません。保存先を確認し、`git status`と`git check-ignore`で公開対象へ混入していないことを確認してください。

