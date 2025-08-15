#!/usr/bin/env python3
"""
ONNXモデルの動画・カメラ推論
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class WaniCalibration:
    """ワニ位置キャリブレーションクラス"""

    def __init__(self, center_x=320, center_y=240, spacing=100, width=80, height=60, config_file="wani_calibration.json"):
        self.center_x = center_x  # 中心位置 X
        self.center_y = center_y  # 中心位置 Y
        self.spacing = spacing  # ワニ間の距離
        self.width = width  # ワニの横幅
        self.height = height  # ワニの縦幅
        self.step_size = 5  # 調整ステップサイズ
        self.config_file = config_file

        # カメラ変換パラメータ
        self.crop_x = 0  # クロップ開始X座標
        self.crop_y = 0  # クロップ開始Y座標
        self.crop_width = 640  # クロップ幅
        self.crop_height = 480  # クロップ高さ
        self.rotation_angle = 0  # 回転角度（度）
        self.crop_step = 10  # クロップ調整ステップ
        self.rotation_step = 1  # 回転調整ステップ

        # 設定ファイルが存在すれば読み込み
        self.load_config()

    def get_wani_zones(self):
        """ワニの5つのゾーンを取得"""
        zones = []
        for i in range(5):
            # 中央を基準に左右に配置 (-2, -1, 0, 1, 2)
            offset_x = (i - 2) * self.spacing
            x1 = self.center_x + offset_x - self.width // 2
            y1 = self.center_y - self.height // 2
            x2 = self.center_x + offset_x + self.width // 2
            y2 = self.center_y + self.height // 2

            zones.append(
                {"id": f"wani_{i + 1:02d}", "bbox": [x1, y1, x2, y2], "center": [self.center_x + offset_x, self.center_y]}
            )

        return zones

    def assign_detection_to_zone(self, detection_bbox):
        """検出バウンディングボックスをゾーンに割り当て"""
        det_x1, det_y1, det_x2, det_y2 = detection_bbox
        det_center_x = (det_x1 + det_x2) / 2
        det_center_y = (det_y1 + det_y2) / 2

        zones = self.get_wani_zones()
        best_zone = None
        min_distance = float("inf")

        for zone in zones:
            zone_x1, zone_y1, zone_x2, zone_y2 = zone["bbox"]

            # バウンディングボックスの重なりをチェック
            overlap_x = max(0, min(det_x2, zone_x2) - max(det_x1, zone_x1))
            overlap_y = max(0, min(det_y2, zone_y2) - max(det_y1, zone_y1))
            overlap_area = overlap_x * overlap_y

            if overlap_area > 0:
                # 重なりがある場合は中心からの距離で判定
                zone_center_x, zone_center_y = zone["center"]
                distance = np.sqrt((det_center_x - zone_center_x) ** 2 + (det_center_y - zone_center_y) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    best_zone = zone["id"]

        return best_zone

    def get_status_text(self):
        """現在のパラメータを文字列で取得"""
        return [
            f"Center: ({self.center_x}, {self.center_y})",
            f"Spacing: {self.spacing}",
            f"Size: {self.width}x{self.height}",
            f"Crop: ({self.crop_x},{self.crop_y}) {self.crop_width}x{self.crop_height}",
            f"Rotation: {self.rotation_angle}°",
        ]

    def save_config(self):
        """設定をファイルに保存"""
        config = {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "spacing": self.spacing,
            "width": self.width,
            "height": self.height,
            "crop_x": self.crop_x,
            "crop_y": self.crop_y,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
            "rotation_angle": self.rotation_angle,
        }
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
            print(f"  設定保存: {self.config_file}")
        except Exception as e:
            print(f"  設定保存エラー: {e}")

    def load_config(self):
        """設定をファイルから読み込み"""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                self.center_x = config.get("center_x", self.center_x)
                self.center_y = config.get("center_y", self.center_y)
                self.spacing = config.get("spacing", self.spacing)
                self.width = config.get("width", self.width)
                self.height = config.get("height", self.height)
                self.crop_x = config.get("crop_x", self.crop_x)
                self.crop_y = config.get("crop_y", self.crop_y)
                self.crop_width = config.get("crop_width", self.crop_width)
                self.crop_height = config.get("crop_height", self.crop_height)
                self.rotation_angle = config.get("rotation_angle", self.rotation_angle)
                print(f"  設定読み込み: {self.config_file}")
        except Exception as e:
            print(f"  設定読み込みエラー: {e}")

    def adjust_parameter(self, param_name, direction):
        """パラメータを調整"""
        if param_name == "center_x":
            self.center_x += direction * self.step_size
        elif param_name == "center_y":
            self.center_y += direction * self.step_size
        elif param_name == "spacing":
            self.spacing = max(20, self.spacing + direction * self.step_size)
        elif param_name == "width":
            self.width = max(20, self.width + direction * self.step_size)
        elif param_name == "height":
            self.height = max(20, self.height + direction * self.step_size)
        elif param_name == "crop_x":
            self.crop_x = max(0, self.crop_x + direction * self.crop_step)
        elif param_name == "crop_y":
            self.crop_y = max(0, self.crop_y + direction * self.crop_step)
        elif param_name == "crop_width":
            self.crop_width = max(100, self.crop_width + direction * self.crop_step)
        elif param_name == "crop_height":
            self.crop_height = max(100, self.crop_height + direction * self.crop_step)
        elif param_name == "rotation":
            self.rotation_angle = (self.rotation_angle + direction * self.rotation_step) % 360

        # 調整後に自動保存
        self.save_config()

    def apply_camera_transform(self, frame):
        """カメラフレームにクロップと回転を適用"""
        if frame is None:
            return frame

        # 回転を適用
        if self.rotation_angle != 0:
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
            frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

        # クロップを適用
        height, width = frame.shape[:2]
        x1 = max(0, min(self.crop_x, width - 100))
        y1 = max(0, min(self.crop_y, height - 100))
        x2 = min(width, x1 + self.crop_width)
        y2 = min(height, y1 + self.crop_height)

        cropped = frame[y1:y2, x1:x2]

        # クロップ後のサイズが元の解像度と異なる場合、リサイズ
        if cropped.shape[0] != height or cropped.shape[1] != width:
            cropped = cv2.resize(cropped, (width, height))

        return cropped


def preprocess_frame(frame, img_size=640):
    """フレームの前処理"""
    if frame is None:
        raise ValueError("フレームがNoneです")

    img = frame.copy()
    original_shape = img.shape[:2]  # H, W

    # リサイズ（アスペクト比を保持）
    scale = img_size / max(original_shape)
    new_shape = (int(original_shape[1] * scale), int(original_shape[0] * scale))
    img_resized = cv2.resize(img, new_shape)

    # パディング
    dw = (img_size - new_shape[0]) / 2
    dh = (img_size - new_shape[1]) / 2
    top, bottom = int(np.round(dh - 0.1)), int(np.round(dh + 0.1))
    left, right = int(np.round(dw - 0.1)), int(np.round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # 正規化とチャンネル順序変更
    img_normalized = img_padded.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
    img_batch = np.expand_dims(img_transposed, axis=0)  # Add batch dimension

    return img_batch, img, scale, (left, top)


def postprocess_predictions(outputs, img_shape, scale, padding, conf_threshold=0.5):
    """予測結果の後処理"""
    predictions = outputs[0]  # Shape: (1, 300, 5) - [x, y, w, h, confidence]

    detections = []
    for pred in predictions[0]:  # 300個の予測
        x, y, w, h, conf = pred

        if conf < conf_threshold:
            continue

        # バウンディングボックスをcenter形式からcorner形式に変換
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # 640x640座標から元画像座標に変換
        # パディングを考慮
        x1 = (x1 * 640 - padding[0]) / scale
        y1 = (y1 * 640 - padding[1]) / scale
        x2 = (x2 * 640 - padding[0]) / scale
        y2 = (y2 * 640 - padding[1]) / scale

        # クリッピング
        x1 = max(0, min(x1, img_shape[1]))
        y1 = max(0, min(y1, img_shape[0]))
        x2 = max(0, min(x2, img_shape[1]))
        y2 = max(0, min(y2, img_shape[0]))

        detections.append({"bbox": [x1, y1, x2, y2], "confidence": float(conf), "class": "wani"})

    return detections


def draw_calibration_zones(img, calibration):
    """キャリブレーションゾーンを描画"""
    zones = calibration.get_wani_zones()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # ゾーンごとの色

    for i, zone in enumerate(zones):
        x1, y1, x2, y2 = [int(coord) for coord in zone["bbox"]]
        color = colors[i % len(colors)]

        # ゾーンの枠を描画（点線）
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # ゾーンIDを描画
        label = zone["id"]
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - label_size[1] - 8), (x1 + label_size[0] + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 中心点を描画
        center_x, center_y = [int(coord) for coord in zone["center"]]
        cv2.circle(img, (center_x, center_y), 3, color, -1)


def draw_detections_with_zones(img, detections, calibration):
    """検出結果とゾーンを描画"""
    img_draw = img.copy()

    # キャリブレーションゾーンを描画
    draw_calibration_zones(img_draw, calibration)

    # 検出結果を描画
    for det in detections:
        x1, y1, x2, y2 = [int(coord) for coord in det["bbox"]]
        conf = det["confidence"]

        # ゾーン割り当て
        zone_id = calibration.assign_detection_to_zone(det["bbox"])

        # バウンディングボックス描画
        if zone_id:
            color = (0, 255, 0)  # ゾーン内の場合は緑
            label = f"{zone_id}: {conf:.2f}"
        else:
            color = (0, 165, 255)  # ゾーン外の場合はオレンジ
            label = f"unknown: {conf:.2f}"

        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

        # ラベル描画
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_draw, (x1, y2 + 2), (x1 + label_size[0] + 4, y2 + label_size[1] + 6), color, -1)
        cv2.putText(img_draw, label, (x1 + 2, y2 + label_size[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img_draw


def draw_detections(img, detections):
    """検出結果を画像に描画（従来版）"""
    img_draw = img.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(coord) for coord in det["bbox"]]
        conf = det["confidence"]

        # バウンディングボックス描画
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ラベル描画
        label = f"wani: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_draw, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(img_draw, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img_draw


def run_frame_inference(session, frame, input_name, output_names, conf_threshold=0.5):
    """単一フレームで推論実行"""
    # フレーム前処理
    img_input, img_original, scale, padding = preprocess_frame(frame)

    # 推論実行
    outputs = session.run(output_names, {input_name: img_input})

    # 後処理
    detections = postprocess_predictions(outputs, img_original.shape[:2], scale, padding, conf_threshold)

    return detections, img_original


def run_camera_inference(
    model_path, camera_id=0, conf_threshold=0.5, record_video=False, fps_limit=30, enable_calibration=False, provider="auto"
):
    """USBカメラでリアルタイム推論実行"""
    print(f"\n📹 USBカメラ: {camera_id}")

    # ONNXセッション作成
    print("📦 ONNXモデルロード中...")

    # プロバイダー選択
    def get_providers(provider_choice):
        available = ort.get_available_providers()

        if provider_choice == "tensorrt":
            if "TensorrtExecutionProvider" not in available:
                print("⚠️  TensorRT未対応、CUDAにフォールバック")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["TensorrtExecutionProvider", "CPUExecutionProvider"]
        elif provider_choice == "cuda":
            if "CUDAExecutionProvider" not in available:
                print("⚠️  CUDA未対応、CPUにフォールバック")
                return ["CPUExecutionProvider"]
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider_choice == "cpu":
            return ["CPUExecutionProvider"]

    providers = get_providers(provider)

    # セッションオプション設定
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # プロバイダー固有のオプション（プロバイダーと同じ数必要）
    provider_options = []

    if "TensorrtExecutionProvider" in providers:
        print("  ⏳ TensorRT初回最適化中... (数分かかります)")
        # 各プロバイダーに対応するオプションを設定
        for provider in providers:
            if provider == "TensorrtExecutionProvider":
                provider_options.append(
                    {
                        "trt_max_workspace_size": "268435456",  # 256MB (最小化)
                        "trt_engine_cache_enable": "True",  # キャッシュ有効
                        "trt_engine_cache_path": "./trt_cache",  # キャッシュパス
                    }
                )
            else:
                provider_options.append({})  # 他のプロバイダー用の空オプション
    elif "CUDAExecutionProvider" in providers:
        print("  ⚡ CUDA起動最適化中...")
        # 各プロバイダーに対応するオプションを設定
        for provider in providers:
            if provider == "CUDAExecutionProvider":
                provider_options.append(
                    {
                        "device_id": 0,  # GPU ID
                        "arena_extend_strategy": "kNextPowerOfTwo",  # メモリ効率化
                        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB制限
                        "cudnn_conv_algo_search": "HEURISTIC",  # 高速アルゴリズム選択
                        "do_copy_in_default_stream": True,  # デフォルトストリーム使用
                    }
                )
            else:
                provider_options.append({})  # 他のプロバイダー用の空オプション
    else:
        # CPUのみの場合
        for provider in providers:
            provider_options.append({})

    session = ort.InferenceSession(str(model_path), sess_options, providers=providers, provider_options=provider_options)

    # 実際に使用されているプロバイダーを表示
    active_provider = session.get_providers()[0]
    print(f"  🚀 使用プロバイダー: {active_provider}")
    if active_provider == "TensorrtExecutionProvider":
        print("  ⚡ TensorRT加速: 最高速度")
    elif active_provider == "CUDAExecutionProvider":
        print("  🔥 CUDA加速: 高速")
    else:
        print("  🖥️  CPU実行: 標準")

    # 入力情報取得
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"  入力: {input_name}, Shape: {input_shape}")

    # 出力情報取得
    output_names = [output.name for output in session.get_outputs()]
    print(f"  出力: {output_names}")

    # カメラ接続
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"USBカメラを開けません: {camera_id}")

    # カメラ設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps_limit)

    # カメラ情報取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  カメラ情報: {width}x{height}, {actual_fps:.1f}fps")

    # 録画設定
    output_writer = None
    if record_video:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"wani_camera_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_writer = cv2.VideoWriter(str(output_path), fourcc, fps_limit, (width, height))
        print(f"📹 録画開始: {output_path}")

    frame_count = 0
    total_detections = 0
    inference_times = []
    fps_control_time = 1.0 / fps_limit

    # キャリブレーション設定（常に作成して設定を読み込み）
    calibration = WaniCalibration(center_x=width // 2, center_y=height // 2)

    def print_calibration_help():
        """キャリブレーション操作方法を表示"""
        print("\n" + "=" * 50)
        print("🎯 キャリブレーションモード操作方法")
        print("=" * 50)
        print("  基本操作:")
        print("    'q': 終了, 's': スクリーンショット, '?': このヘルプを再表示")
        print("  === ワニゾーン調整 ===")
        print("    'w/a/s/d': 中心位置調整")
        print("    'i/k': 間隔調整, 'j/l': 横幅調整, 'u/o': 縦幅調整")
        print("  === カメラ調整 ===")
        print("    'y/h': クロップ位置上下, 'g/f': クロップ位置左右")
        print("    '1/2': クロップ幅調整, '3/4': クロップ高さ調整")
        print("    'r/t': 回転調整")
        print("=" * 50)

    if enable_calibration:
        print_calibration_help()
    else:
        print("⚡ リアルタイム推論開始...")
        print("  'q'キーで終了, 's'キーでスクリーンショット保存")
        print("  キャリブレーション枠を表示します")

    try:
        last_time = time.time()

        while True:
            # FPS制御（キャリブレーション時は無効）
            if not enable_calibration:
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed < fps_control_time:
                    time.sleep(fps_control_time - elapsed)

            ret, frame = cap.read()
            if not ret:
                print("⚠️ フレームの取得に失敗")
                continue

            # カメラ変換を適用（クロップ・回転）
            frame = calibration.apply_camera_transform(frame)

            frame_count += 1
            if not enable_calibration:
                last_time = time.time()

            # キャリブレーション時は推論をスキップ
            detections = []
            if not enable_calibration:
                # 推論実行
                start_time = time.time()
                detections, processed_frame = run_frame_inference(session, frame, input_name, output_names, conf_threshold)
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
                total_detections += len(detections)

            # 検出結果描画
            if enable_calibration:
                # キャリブレーションモード：枠のみ表示
                frame_with_detections = frame.copy()
                draw_calibration_zones(frame_with_detections, calibration)
            else:
                # 通常モード：検出結果 + キャリブレーション枠を表示
                if detections:
                    frame_with_detections = draw_detections_with_zones(frame, detections, calibration)
                else:
                    frame_with_detections = frame.copy()
                    draw_calibration_zones(frame_with_detections, calibration)

            # FPSと統計情報表示
            info_text = []
            if not enable_calibration and len(inference_times) > 0:
                avg_inference_time = np.mean(inference_times[-30:])
                current_fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
                info_text = [
                    f"FPS: {current_fps:.1f}",
                    f"Inference: {avg_inference_time:.1f}ms",
                    f"Detections: {len(detections)}",
                    f"Total: {total_detections}",
                ]

            # キャリブレーション情報を追加
            if enable_calibration:
                info_text.extend(calibration.get_status_text())
                info_text.insert(0, "CALIBRATION MODE")

            # 情報を画面に表示
            if info_text:
                y_offset = 30
                for i, text in enumerate(info_text):
                    cv2.putText(
                        frame_with_detections, text, (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

            # 画面表示
            cv2.imshow("Wani Camera Detection", frame_with_detections)

            # ウィンドウにフォーカスを設定（キー入力を確実に受け取るため）
            cv2.setWindowProperty("Wani Camera Detection", cv2.WND_PROP_TOPMOST, 1)
            cv2.setWindowProperty("Wani Camera Detection", cv2.WND_PROP_TOPMOST, 0)

            # 録画
            if output_writer is not None:
                output_writer.write(frame_with_detections)

            # キー入力処理（応答性とフレームレートのバランス）
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                print("\n⏹️ ユーザーが終了")
                break
            elif key == ord("s") and not enable_calibration:
                # スクリーンショット（キャリブレーション時は's'が中心Y調整に使用）
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"wani_screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, frame_with_detections)
                print(f"  📷 スクリーンショット保存: {screenshot_path}")
            elif enable_calibration:
                # キャリブレーション用キー操作
                if key == ord("w"):  # 中心Y上
                    calibration.adjust_parameter("center_y", -1)
                    print(f"  中心Y上移動: Y={calibration.center_y}")
                elif key == ord("s"):  # 中心Y下
                    calibration.adjust_parameter("center_y", 1)
                    print(f"  中心Y下移動: Y={calibration.center_y}")
                elif key == ord("a"):  # 中心X左
                    calibration.adjust_parameter("center_x", -1)
                    print(f"  中心X左移動: X={calibration.center_x}")
                elif key == ord("d"):  # 中心X右
                    calibration.adjust_parameter("center_x", 1)
                    print(f"  中心X右移動: X={calibration.center_x}")
                elif key == ord("i"):  # 間隔拡大
                    calibration.adjust_parameter("spacing", 1)
                    print(f"  間隔拡大: {calibration.spacing}")
                elif key == ord("k"):  # 間隔縮小
                    calibration.adjust_parameter("spacing", -1)
                    print(f"  間隔縮小: {calibration.spacing}")
                elif key == ord("j"):  # 横幅縮小
                    calibration.adjust_parameter("width", -1)
                    print(f"  横幅縮小: {calibration.width}")
                elif key == ord("l"):  # 横幅拡大
                    calibration.adjust_parameter("width", 1)
                    print(f"  横幅拡大: {calibration.width}")
                elif key == ord("u"):  # 縦幅縮小
                    calibration.adjust_parameter("height", -1)
                    print(f"  縦幅縮小: {calibration.height}")
                elif key == ord("o"):  # 縦幅拡大
                    calibration.adjust_parameter("height", 1)
                    print(f"  縦幅拡大: {calibration.height}")
                # カメラ調整キー（通常キー）
                elif key == ord("r"):  # 反時計回り回転
                    calibration.adjust_parameter("rotation", -1)
                    print(f"  反時計回り回転: {calibration.rotation_angle}°")
                elif key == ord("t"):  # 時計回り回転
                    calibration.adjust_parameter("rotation", 1)
                    print(f"  時計回り回転: {calibration.rotation_angle}°")
                elif key == ord("y"):  # クロップY位置上
                    calibration.adjust_parameter("crop_y", -1)
                    print(f"  クロップ位置上: Y={calibration.crop_y}")
                elif key == ord("h"):  # クロップY位置下
                    calibration.adjust_parameter("crop_y", 1)
                    print(f"  クロップ位置下: Y={calibration.crop_y}")
                elif key == ord("g"):  # クロップX位置左
                    calibration.adjust_parameter("crop_x", -1)
                    print(f"  クロップ位置左: X={calibration.crop_x}")
                elif key == ord("f"):  # クロップX位置右
                    calibration.adjust_parameter("crop_x", 1)
                    print(f"  クロップ位置右: X={calibration.crop_x}")
                elif key == ord("1"):  # クロップ幅縮小
                    calibration.adjust_parameter("crop_width", -1)
                    print(f"  クロップ幅縮小: {calibration.crop_width}")
                elif key == ord("2"):  # クロップ幅拡大
                    calibration.adjust_parameter("crop_width", 1)
                    print(f"  クロップ幅拡大: {calibration.crop_width}")
                elif key == ord("3"):  # クロップ高さ縮小
                    calibration.adjust_parameter("crop_height", -1)
                    print(f"  クロップ高さ縮小: {calibration.crop_height}")
                elif key == ord("4"):  # クロップ高さ拡大
                    calibration.adjust_parameter("crop_height", 1)
                    print(f"  クロップ高さ拡大: {calibration.crop_height}")
                # ヘルプ表示
                elif key == ord("?"):  # ヘルプ表示
                    print_calibration_help()
                # デバッグ用：どのキーが押されたかを表示
                elif key != 255:  # 255は何もキーが押されていない状態
                    if 32 <= key <= 126:  # 印刷可能文字の場合
                        print(f"  キー押下: '{chr(key)}' (コード: {key})")
                    else:
                        print(f"  キー押下: 特殊キー (コード: {key})")

            # 進捗表示（コンソール）
            if frame_count % 100 == 0:
                avg_inference_time = np.mean(inference_times[-100:])
                avg_detections_per_frame = total_detections / frame_count
                print(
                    f"  フレーム: {frame_count}, 平均推論: {avg_inference_time:.1f}ms, "
                    f"平均検出: {avg_detections_per_frame:.2f}"
                )

    finally:
        cap.release()
        if output_writer is not None:
            output_writer.release()
        cv2.destroyAllWindows()

    # 統計情報表示
    if len(inference_times) > 0:
        avg_inference_time = np.mean(inference_times)
        avg_detections_per_frame = total_detections / frame_count if frame_count > 0 else 0

        print("\n✅ カメラ推論終了!")
        print(f"  処理フレーム数: {frame_count}")
        print(f"  総検出数: {total_detections}")
        print(f"  平均検出数/フレーム: {avg_detections_per_frame:.2f}")
        print(f"  平均推論時間: {avg_inference_time:.1f}ms")
        print(f"  実効FPS: {1000 / avg_inference_time:.1f}fps")

    return total_detections


def main():
    parser = argparse.ArgumentParser(description="ONNX動画・カメラ推論")
    parser.add_argument("--model", type=str, default="models/wani_detector.onnx", help="ONNXモデルパス")
    parser.add_argument("--camera", type=int, help="カメラID（通常0）- USBカメラ使用時")
    parser.add_argument("--conf", type=float, default=0.5, help="信頼度閾値")
    parser.add_argument("--record", action="store_true", help="カメラ映像を録画（カメラのみ）")
    parser.add_argument("--fps", type=int, default=30, help="カメラFPS制限")
    parser.add_argument("--calibrate", action="store_true", help="5匹ワニ位置キャリブレーションモード")
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["tensorrt", "cuda", "cpu"],
        help="実行プロバイダー: cpu(CPU・デフォルト), cuda(CUDA), tensorrt(TensorRT)",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("🚀 ONNX Runtime推論")
    print("=" * 50)

    # モデル存在確認
    if not Path(args.model).exists():
        print(f"❌ モデルが見つかりません: {args.model}")
        return

    # 入力モード判定
    if args.camera is not None:
        # カメラモード
        print(f"📷 USBカメラモード: Camera ID {args.camera}")
        run_camera_inference(
            args.model,
            camera_id=args.camera,
            conf_threshold=args.conf,
            record_video=args.record,
            fps_limit=args.fps,
            enable_calibration=args.calibrate,
            provider=args.provider,
        )
    # ファイルモードは現在サポートされていません

    else:
        print("❌ --camera を指定してください")
        print("\n使用例:")
        print("  カメラ: python wani_detector.py --camera 0 --record")
        print("  キャリブレーション: python wani_detector.py --camera 0 --calibrate")
        parser.print_help()


if __name__ == "__main__":
    main()
