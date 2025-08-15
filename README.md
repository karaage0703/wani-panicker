# Wani Panicker - ワニ叩きロボットシステム

このリポジトリは[LeRobot](https://github.com/huggingface/lerobot)を使用したワニ検出・自動叩きロボットシステムです。

> **Note**: このプロジェクトは[Hugging Face LeRobot](https://github.com/huggingface/lerobot)プロジェクトを基盤としています。LeRobotは Apache-2.0 ライセンスの下で公開されています。

## 機能

- **Wani Panicker**: カメラでワニを検出し、自動でロボットアームが叩くシステム
- **Motion Editor**: キーボード操作によるポイントtoポイントモーション作成・編集ツール
- **Wani Player**: キーボードでワニモーション00-05を手動再生するツール
- **Wani Detector**: ONNXモデルによるワニ検出システム（キャリブレーション機能付き）
- uvを使用した仮想環境管理
- ruffによるコード品質管理

## セットアップ

### 1. uvによる環境構築

```bash
uv sync
```

これにより以下がインストールされます：
- lerobot
- lerobot[feetech] (Feetech Servo SDK含む)
- opencv-python (画像処理・カメラ制御)
- onnxruntime (AI推論)
- numpy (数値計算)
- draccus (設定管理)
- pynput (キーボード制御)
- ruff (開発依存関係)

### 2. 仮想環境の有効化

```bash
source .venv/bin/activate
```

または

```bash
uv run <command>
```

## Wani Panicker

カメラでワニを検出し、自動でロボットアームが対応するモーションで叩くメインシステムです。

### 使用方法

```bash
uv run wani_panicker.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/ttyUSB0
```

### 動作説明

1. カメラでフレームを取得
2. ONNXモデルでワニ検出
3. 検出されたワニをキャリブレーション済みの5つのゾーン（wani_01〜wani_05）に割り当て
4. 対応するモーションファイル（motion_wani_01.json〜motion_wani_05.json）を自動実行
5. ロボットアームがワニを叩く

### オプション

- `--model-path`: ONNXモデルファイルのパス（デフォルト: models/wani_detector.onnx）
- `--camera-id`: カメラID（デフォルト: 0）
- `--conf-threshold`: 検出信頼度閾値（デフォルト: 0.5）
- `--speed`: モーション再生速度倍率（デフォルト: 0.8）
- `--detection-cooldown`: 同じワニへの連続検出を防ぐ時間（秒、デフォルト: 3.0）
- `--fps-limit`: カメラFPS制限（デフォルト: 10）
- `--verbose`: デバッグ情報表示

### 必要ファイル

- `models/wani_detector.onnx`: ワニ検出用AIモデル
- `motions/motion_wani_01.json`〜`motions/motion_wani_05.json`: 各ワニ位置用モーションファイル
- `wani_calibration.json`: キャリブレーション設定（自動生成）

## Wani Detector

ONNXモデルを使用したワニ検出システムです。キャリブレーション機能により、カメラの位置調整とワニエリアの設定が可能です。

### 使用方法

#### 基本的な検出

```bash
uv run wani_detector.py --camera 0
```

#### キャリブレーションモード

```bash
uv run wani_detector.py --camera 0 --calibrate
```

### キャリブレーション操作

#### ワニゾーン調整
- `w/a/s/d`: 中心位置調整
- `i/k`: 間隔調整
- `j/l`: 横幅調整
- `u/o`: 縦幅調整

#### カメラ調整
- `y/h`: クロップ位置上下
- `g/f`: クロップ位置左右
- `1/2`: クロップ幅調整
- `3/4`: クロップ高さ調整
- `r/t`: 回転調整

#### 基本操作
- `q`: 終了
- `s`: スクリーンショット保存
- `?`: ヘルプ表示

## Wani Player

キーボードでワニモーション00-05を手動再生するツールです。

### 使用方法

```bash
uv run wani_player.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/ttyUSB0
```

### キーボード操作

- `0-5`: 対応するワニモーション（motion_wani_00.json〜motion_wani_05.json）を再生
- `h`: ホームポジションに移動
- `ESC`: 終了

## Motion Editor

キーボード操作でポイントtoポイントモーションを作成・編集できるツールです。

### 使用方法

```bash
uv run python motion_editor.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/ttyUSB0
```

### キーボードコマンド

- **WASD**: 各軸の移動 (W/S: shoulder_pan, A/D: shoulder_lift)
- **IJKL**: その他の軸 (I/K: elbow_flex, J/L: wrist_flex)
- **Q/E**: wrist_roll
- **Z/X**: gripper
- **M**: 現在位置をポイントとして記録
- **P**: 記録されたモーションを再生
- **S**: モーションをファイルに保存
- **L**: モーションをファイルから読み込み
- **R**: モーションをリセット（2回押しで確定）
- **ESC**: 終了

## 開発

### コード品質チェック

```bash
# Lintチェック
uv run ruff check

# 自動修正
uv run ruff check --fix

# フォーマット
uv run ruff format
```

### 設定

ruffの設定は`pyproject.toml`で管理されています：
- line-length: 127
- 基本的なlintルール (E, F, W, I) を有効化

## プロジェクト構造

```
.
├── wani_panicker.py       # メインシステム：ワニ検出・自動叩きシステム
├── wani_detector.py       # ワニ検出システム（キャリブレーション機能付き）
├── wani_player.py         # ワニモーション手動再生ツール
├── motion_editor.py       # Motion Editorメインファイル
├── motion_utils.py        # Motion関連共通ユーティリティ
├── wani_calibration.json  # キャリブレーション設定（自動生成）
├── models/                # AIモデルファイル
│   └── wani_detector.onnx   # ワニ検出用ONNXモデル（要配置）
├── motions/               # 保存されたモーションファイル
├── pyproject.toml         # uv/ruff設定
├── .venv/                # 仮想環境 (uv syncで自動作成)
└── README.md             # このファイル
```

## システム全体フロー

1. **モーション作成**: `motion_editor.py`でワニを叩くモーションを作成・保存
2. **キャリブレーション**: `wani_detector.py --calibrate`でカメラとワニエリアを調整
3. **テスト**: `wani_player.py`で各モーションが正常に動作することを確認
4. **自動運用**: `wani_panicker.py`でワニ検出・自動叩きシステムを起動

## ライセンス

このプロジェクト自体はMITライセンスの下で公開されています。

### 使用しているプロジェクト

- [LeRobot](https://github.com/huggingface/lerobot): Apache-2.0 ライセンス
- 本プロジェクトで作成したコード: MIT ライセンス

LeRobotライブラリを使用する場合は、Apache-2.0ライセンスの条項に従ってください。