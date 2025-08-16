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

### 0. USBシリアルデバイス設定（必須）

ロボットの接続を安定化するため、udev rulesによるUSBシリアルデバイスの設定を行います。

#### デバイス情報の確認

```bash
# USBデバイス一覧表示
lsusb

# 特定デバイスのシリアル番号確認（例：1a86:55d3）
lsusb -v -s [bus:device] | grep -i serial
```

#### udev rulesファイルの作成

```bash
# udev rulesファイルを作成
sudo cp config/99-lerobot-serial.rules /etc/udev/rules.d/

# または手動で作成
sudo nano /etc/udev/rules.d/99-lerobot-serial.rules
```

**99-lerobot-serial.rules の内容例：**
```
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5AA9018069", SYMLINK+="usbserial_lerobot_follower", MODE="0666"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5AA9017941", SYMLINK+="usbserial_lerobot_leader", MODE="0666"
```

#### udev rulesの適用

```bash
# udev rulesをリロード
sudo udevadm control --reload

# デバイスを再接続または以下でトリガー
sudo udevadm trigger
```

#### 設定確認

```bash
# シンボリックリンクが作成されているか確認
ls -la /dev/usbserial_*

# 期待される出力例：
# lrwxrwxrwx 1 root root 7 Aug 16 15:30 /dev/usbserial_lerobot_follower -> ttyUSB0
```

この設定により、USB接続順序に関係なく`/dev/usbserial_lerobot_follower`で一貫してロボットにアクセスできます。

> **参考**: [USB シリアル通信デバイスを udev で管理する | Zenn](https://zenn.dev/karaage0703/articles/8042463b476fbf)

## セットアップ

### 1. 標準環境（PC・開発環境）

#### uvによる環境構築

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

#### 仮想環境の有効化

```bash
source .venv/bin/activate
```

または

```bash
uv run <command>
```

### 2. Jetson Orin環境（高速推論対応）

Jetson Orin用の最適化されたセットアップスクリプトを実行します。

> **重要**: Jetson Orin環境では、NVIDIA提供のonnxruntime-gpu wheelファイルがPyPIにないため、**uvは使用せずシステムのPython環境**を直接使用します。uvでは外部wheelファイルの直接インストールに制限があるためです。

#### 統合セットアップスクリプトの実行

```bash
# 実行権限を付与
chmod +x setup_jetson_lerobot.sh
chmod +x setup_jetson_onnxruntime-gpu.sh

# Step 1: LeRobotとPython依存関係のセットアップ
./setup_jetson_lerobot.sh

# Step 2: ONNX Runtime GPU版のセットアップ
./setup_jetson_onnxruntime-gpu.sh
```

#### setup_jetson_lerobot.sh の内容
このスクリプトにより以下が自動で実行されます：
- システム情報確認
- Python依存関係の競合解決
- LeRobot[feetech]のインストール
- NumPy 2.0互換性問題の回避

#### setup_jetson_onnxruntime-gpu.sh の内容  
このスクリプトにより以下が自動で実行されます：
- JetPack 6.0対応ONNX Runtime GPU版の自動ダウンロード
- 既存ONNX Runtimeの削除とGPU版インストール
- TensorRT・CUDAプロバイダーの動作確認
- 自動クリーンアップ

#### 手動での依存関係解決（必要な場合）

```bash
# システムのpandas/numpyとの競合解決
sudo apt remove python3-pandas python3-numpy

# NumPy 2.0互換性問題の回避
pip3 install "numpy<2.0" --force-reinstall

# OpenCVのGUI対応版を優先
pip3 uninstall -y opencv-python-headless
pip3 install opencv-python
```

#### Jetson環境での実行

```bash
# CPU実行（デフォルト・安定）
python3 wani_detector.py --camera 0

# CUDA実行（高速・推奨）
python3 wani_detector.py --camera 0 --provider cuda

# TensorRT実行（最高速・初回時間かかる）
python3 wani_detector.py --camera 0 --provider tensorrt
```

> **Note**: 
> - Jetson環境では`python3`コマンドを直接使用（uv runは使用しない）
> - TensorRTはまだ動作未確認です

## Wani Panicker

カメラでワニを検出し、自動でロボットアームが対応するモーションで叩くメインシステムです。

### 使用方法

```bash
# 標準環境（PC・開発環境）
uv run wani_panicker.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/usbserial_lerobot_follower
# または
source .venv/bin/activate && python3 wani_panicker.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/usbserial_lerobot_follower

# Jetson Orin環境（システムPython使用）
python3 wani_panicker.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/usbserial_lerobot_follower

# udev rules未設定の場合は /dev/ttyUSB0 を使用
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
# 標準環境（PC・開発環境）
uv run wani_detector.py --camera 0
# または
source .venv/bin/activate && python3 wani_detector.py --camera 0

# Jetson Orin環境（システムPython使用）
python3 wani_detector.py --camera 0
```

#### キャリブレーションモード

```bash
# 標準環境（PC・開発環境）
uv run wani_detector.py --camera 0 --calibrate
# または
source .venv/bin/activate && python3 wani_detector.py --camera 0 --calibrate

# Jetson Orin環境（システムPython使用）
python3 wani_detector.py --camera 0 --calibrate
```

### オプション

- `--model`: ONNXモデルファイルのパス（デフォルト: models/wani_detector.onnx）
- `--camera`: カメラID（必須、通常は0）
- `--conf`: 検出信頼度閾値（デフォルト: 0.5）
- `--record`: カメラ映像を録画（カメラモードのみ）
- `--fps`: カメラFPS制限（デフォルト: 30）
- `--calibrate`: キャリブレーションモードを有効化
- `--provider`: 実行プロバイダー（デフォルト: cpu）
  - `cpu`: CPU実行（安定・デフォルト）
  - `cuda`: CUDA実行（高速・Jetson推奨）
  - `tensorrt`: TensorRT実行（最高速・Jetson専用）

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
# 標準環境（PC・開発環境）
uv run wani_player.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/usbserial_lerobot_follower
# または
source .venv/bin/activate && python3 wani_player.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/usbserial_lerobot_follower

# Jetson Orin環境（システムPython使用）
python3 wani_player.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/usbserial_lerobot_follower

# udev rules未設定の場合は /dev/ttyUSB0 を使用
```

### キーボード操作

- `0-5`: 対応するワニモーション（motion_wani_00.json〜motion_wani_05.json）を再生
- `h`: ホームポジションに移動
- `ESC`: 終了

## Motion Editor

キーボード操作でポイントtoポイントモーションを作成・編集できるツールです。

### 使用方法

```bash
# 標準環境（PC・開発環境）
uv run motion_editor.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/usbserial_lerobot_follower
# または
source .venv/bin/activate && python3 motion_editor.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/usbserial_lerobot_follower

# Jetson Orin環境（システムPython使用）
python3 motion_editor.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/usbserial_lerobot_follower

# udev rules未設定の場合は /dev/ttyUSB0 を使用
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
├── wani_panicker.py              # メインシステム：ワニ検出・自動叩きシステム
├── wani_detector.py              # ワニ検出システム（キャリブレーション機能付き）
├── wani_player.py                # ワニモーション手動再生ツール
├── motion_editor.py              # Motion Editorメインファイル
├── motion_utils.py               # Motion関連共通ユーティリティ
├── setup_jetson_lerobot.sh       # Jetson Orin用LeRobotセットアップスクリプト
├── setup_jetson_onnxruntime-gpu.sh # Jetson Orin用ONNX Runtime GPUセットアップスクリプト
├── rules.md                      # プロジェクトルール・ガイドライン
├── pyproject.toml                # uv/ruff設定
├── uv.lock                       # uvの依存関係ロックファイル
├── wani_calibration.json         # キャリブレーション設定（自動生成）
├── .venv/                        # 仮想環境 (uv syncで自動作成)
├── .github/                      # GitHub設定
├── config/                       # 設定ファイル
│   └── 99-lerobot-serial.rules     # udev rules（USBシリアルデバイス設定）
├── models/                       # AIモデルファイル
│   └── wani_detector.onnx          # ワニ検出用ONNXモデル（要配置）
├── motions/                      # 保存されたモーションファイル
└── README.md                     # このファイル
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