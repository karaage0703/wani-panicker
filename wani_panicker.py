#!/usr/bin/env python3

"""
Wani Panicker - ワニを検出したら自動でアームで叩くシステム

使用方法:
    python wani_panicker.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/ttyUSB0

機能:
    - カメラでワニ検出
    - 検出されたワニの番号に応じて対応するモーションを自動実行
    - wani_01 → motion_wani_01.json を実行
    - wani_02 → motion_wani_02.json を実行 など
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import draccus
import numpy as np
import onnxruntime as ort
from lerobot.robots import (
    RobotConfig,
    make_robot_from_config,
    so101_follower,  # noqa: F401
)
from lerobot.utils.utils import init_logging

from motion_utils import apply_pid_settings, get_home_position, load_motion_from_file
from wani_detector import (
    WaniCalibration,
    draw_detections_with_zones,
    postprocess_predictions,
    preprocess_frame,
)


@dataclass
class WaniPanickerConfig:
    robot: RobotConfig
    # ONNXモデルパス
    model_path: str = "./models/wani_detector.onnx"
    # カメラID
    camera_id: int = 0
    # 検出信頼度閾値
    conf_threshold: float = 0.5
    # モーションファイルディレクトリ
    motion_dir: str = "./motions"
    # 再生速度の倍率 (1.0が通常速度)
    speed: float = 0.8
    # デバッグ情報を表示するか
    verbose: bool = False
    # 開始時にホームポジションに移動するか
    go_to_home: bool = True
    # ホーム移動の時間（秒）
    home_duration: float = 3.0
    # モーション終了後にホームポジションに戻るか
    return_to_home: bool = True
    # モーション終了後の待機時間（秒）
    end_hold_time: float = 1.0
    # 最適化PID設定を適用するか
    use_optimized_pid: bool = True
    # 同じワニに対する連続検出を無視する時間（秒）
    detection_cooldown: float = 3.0
    # FPS制限
    fps_limit: int = 10


class WaniDetector:
    """ワニ検出クラス"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNXモデルが見つかりません: {model_path}")

        # ONNXセッション作成
        self.session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        print(f"📦 ONNXモデル読み込み完了: {self.model_path}")
        print(f"  入力: {self.input_name}")
        print(f"  出力: {self.output_names}")

    def detect(self, frame):
        """単一フレームでワニ検出を実行"""
        # フレーム前処理
        img_input, img_original, scale, padding = preprocess_frame(frame)

        # 推論実行
        outputs = self.session.run(self.output_names, {self.input_name: img_input})

        # 後処理
        detections = postprocess_predictions(outputs, img_original.shape[:2], scale, padding, self.conf_threshold)

        return detections


class WaniPanicker:
    """ワニパニッカー本体"""

    def __init__(self, cfg: WaniPanickerConfig):
        self.cfg = cfg
        self.robot = None
        self.detector = None
        self.calibration = None
        self.should_exit = False
        self.motions: Dict[str, object] = {}
        self.home_motion = None  # ホームポジション用モーション
        self.is_playing_motion = False
        self.last_detection_time: Dict[str, float] = {}

        # ログレベル設定
        if cfg.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # ワニ検出器を初期化
        self.detector = WaniDetector(cfg.model_path, cfg.conf_threshold)

        # ロボット接続
        print(f"🤖 ロボットに接続中... (type: {cfg.robot.type})")
        self.robot = make_robot_from_config(cfg.robot)
        self.robot.connect()
        print("✅ ロボット接続完了")

        # PID設定を適用
        if cfg.use_optimized_pid:
            apply_pid_settings(self.robot, use_optimized=True)
        else:
            apply_pid_settings(self.robot, use_optimized=False)

        # モーションファイルを事前に読み込み（ホームモーションを含む）
        self._load_all_motions()

        # ホームポジションに移動
        if cfg.go_to_home:
            self._go_to_home()

        # キャリブレーション設定を読み込み
        self.calibration = WaniCalibration()

    def _go_to_home(self):
        """ロボットをホームポジションに移動（motion_wani_00.jsonを使用）"""
        if not self.home_motion:
            print("⚠️  motion_wani_00.jsonが読み込まれていません - 従来のホームポジションを使用")
            # フォールバック: 従来のホームポジション
            home_position = get_home_position()
            try:
                self.robot.send_action(home_position)
                print(f"⏱️  ホーム移動時間: {self.cfg.home_duration}秒")
                time.sleep(self.cfg.home_duration)
                print("✅ ホームポジション到達")
            except Exception as e:
                print(f"❌ ホーム移動エラー: {e}")
            return

        print("🏠 ホームポジションに移動中（motion_wani_00.jsonを実行）...")

        try:
            if not self.home_motion.points:
                print("❌ ホームモーションにポイントがありません")
                return

            for i, point in enumerate(self.home_motion.points, 1):
                if self.cfg.verbose:
                    print(f"📍 ホームポイント {i}/{len(self.home_motion.points)}: {point.name}")
                    for joint, pos in point.positions.items():
                        print(f"     {joint}: {pos:.2f}")

                self.robot.send_action(point.positions)
                time.sleep(point.duration / self.cfg.speed)
                time.sleep(0.1)

            print("✅ ホームポジション到達")

        except Exception as e:
            print(f"❌ ホーム移動エラー: {e}")

    def _load_all_motions(self):
        """全てのワニモーションファイルを事前に読み込み"""
        print("📚 ワニモーションファイルを読み込み中...")

        for i in range(0, 6):  # 0-5 (wani_00 to wani_05)
            motion_file = f"motion_wani_{i:02d}.json"
            motion_path = Path(self.cfg.motion_dir) / motion_file

            print(f"🔍 チェック中: {motion_path}")  # デバッグ情報
            if motion_path.exists():
                try:
                    motion = load_motion_from_file(str(motion_path))
                    if i == 0:
                        # wani_00はホームポジション用として特別に保存
                        self.home_motion = motion
                        print(f"✅ {motion_file}: {motion.name} (ホームポジション)")
                    else:
                        self.motions[f"wani_{i:02d}"] = motion
                        print(f"✅ {motion_file}: {motion.name}")
                except Exception as e:
                    print(f"❌ {motion_file} 読み込みエラー: {e}")
            else:
                if i == 0:
                    print(f"⚠️  {motion_file} が見つかりません（ホームポジション用）")
                    print(f"     パス: {motion_path}")  # デバッグ情報
                else:
                    print(f"⚠️  {motion_file} が見つかりません")

        print(f"📖 読み込み完了: {len(self.motions)}個のモーション")

    def _should_trigger_motion(self, zone_id: str) -> bool:
        """モーションを実行すべきかどうかを判定（連続検出防止）"""
        current_time = time.time()

        if zone_id in self.last_detection_time:
            elapsed = current_time - self.last_detection_time[zone_id]
            if elapsed < self.cfg.detection_cooldown:
                return False

        self.last_detection_time[zone_id] = current_time
        return True

    def _play_motion(self, motion, zone_id: str):
        """モーションを再生（非同期）"""

        def play_motion_thread():
            if not motion.points:
                print("❌ モーションポイントがありません")
                return

            self.is_playing_motion = True
            print(f"🎯 {zone_id}のワニを叩きます！")
            print(f"▶️  モーション再生開始: {motion.name}")

            try:
                for i, point in enumerate(motion.points, 1):
                    print(f"📍 ポイント {i}/{len(motion.points)}: {point.name}")

                    if self.cfg.verbose:
                        for joint, pos in point.positions.items():
                            print(f"     {joint}: {pos:.2f}")

                    self.robot.send_action(point.positions)
                    time.sleep(point.duration / self.cfg.speed)
                    time.sleep(0.1)

                print("✅ ワニ叩き完了！")

                # モーション終了後の処理
                if self.cfg.return_to_home:
                    print(f"⏱️  終了位置で{self.cfg.end_hold_time}秒待機...")
                    time.sleep(self.cfg.end_hold_time)
                    print("🏠 ホームポジションに戻り中...")
                    self._go_to_home()

            except Exception as e:
                print(f"❌ モーション再生エラー: {e}")
            finally:
                self.is_playing_motion = False

        # 別スレッドでモーション実行
        motion_thread = threading.Thread(target=play_motion_thread, daemon=True)
        motion_thread.start()

    def _process_detections(self, detections):
        """検出結果を処理してモーションを実行"""
        if self.is_playing_motion:
            return  # モーション実行中は新しい検出を無視

        for det in detections:
            # ゾーンに割り当て
            zone_id = self.calibration.assign_detection_to_zone(det["bbox"])

            if zone_id and zone_id in self.motions:
                # 連続検出防止チェック
                if self._should_trigger_motion(zone_id):
                    print(f"🚨 {zone_id}でワニを検出！")
                    self._play_motion(self.motions[zone_id], zone_id)
                    break  # 1フレームで1つのワニのみ処理
                else:
                    if self.cfg.verbose:
                        elapsed = time.time() - self.last_detection_time.get(zone_id, 0)
                        remaining = self.cfg.detection_cooldown - elapsed
                        print(f"🕐 {zone_id}: クールダウン中 (残り{remaining:.1f}秒)")

    def run(self):
        """メイン実行"""
        # カメラ接続
        cap = cv2.VideoCapture(self.cfg.camera_id)
        if not cap.isOpened():
            raise ValueError(f"カメラを開けません: {self.cfg.camera_id}")

        # カメラ設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.fps_limit)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 カメラ起動完了: {width}x{height}")
        print("\n🚨 ワニパニッカー開始！")
        print("=== 動作モード ===")
        print("ワニを検出すると自動でアームが叩きます")
        print("'q'キーで終了")
        print("==================")

        frame_count = 0
        total_detections = 0
        inference_times = []
        fps_control_time = 1.0 / self.cfg.fps_limit

        try:
            last_time = time.time()

            while not self.should_exit:
                # FPS制御
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed < fps_control_time:
                    time.sleep(fps_control_time - elapsed)

                ret, frame = cap.read()
                if not ret:
                    print("⚠️ フレームの取得に失敗")
                    continue

                # カメラ変換を適用（クロップ・回転）
                frame = self.calibration.apply_camera_transform(frame)

                frame_count += 1
                last_time = time.time()

                # ワニ検出実行
                start_time = time.time()
                detections = self.detector.detect(frame)
                inference_time = (time.time() - start_time) * 1000
                inference_times.append(inference_time)
                total_detections += len(detections)

                # 検出結果処理
                if detections:
                    self._process_detections(detections)

                # 描画
                frame_with_detections = draw_detections_with_zones(frame, detections, self.calibration)

                # 統計情報表示
                if len(inference_times) > 0:
                    avg_inference_time = np.mean(inference_times[-30:])
                    current_fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0

                    info_text = [
                        f"FPS: {current_fps:.1f}",
                        f"Inference: {avg_inference_time:.1f}ms",
                        f"Detections: {len(detections)}",
                        f"Motion: {'実行中' if self.is_playing_motion else '待機中'}",
                    ]

                    # 情報を画面に表示
                    y_offset = 30
                    for i, text in enumerate(info_text):
                        cv2.putText(
                            frame_with_detections, text, (10, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )

                # 画面表示
                cv2.imshow("Wani Panicker", frame_with_detections)

                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n⏹️ 終了します")
                    self.should_exit = True
                    break

                # 進捗表示
                if frame_count % 100 == 0:
                    avg_detections_per_frame = total_detections / frame_count
                    print(f"  フレーム: {frame_count}, 平均検出: {avg_detections_per_frame:.2f}")

        finally:
            cap.release()
            cv2.destroyAllWindows()

            # ロボット切断
            if self.robot and hasattr(self.robot, "is_connected") and self.robot.is_connected:
                try:
                    self.robot.disconnect()
                    print("🔌 ロボットから切断しました")
                except Exception as e:
                    print(f"⚠️ ロボット切断時にエラー: {e}")

        # 統計情報表示
        if len(inference_times) > 0:
            avg_inference_time = np.mean(inference_times)
            avg_detections_per_frame = total_detections / frame_count if frame_count > 0 else 0

            print("\n✅ ワニパニッカー終了!")
            print(f"  処理フレーム数: {frame_count}")
            print(f"  総検出数: {total_detections}")
            print(f"  平均検出数/フレーム: {avg_detections_per_frame:.2f}")
            print(f"  平均推論時間: {avg_inference_time:.1f}ms")


@draccus.wrap()
def main(cfg: WaniPanickerConfig):
    """メイン関数"""
    init_logging()

    # モーションディレクトリ作成
    Path(cfg.motion_dir).mkdir(exist_ok=True)

    panicker = WaniPanicker(cfg)
    panicker.run()


if __name__ == "__main__":
    main()
