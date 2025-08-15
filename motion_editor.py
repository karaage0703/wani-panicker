#!/usr/bin/env python3

"""
Motion Editor - キーボード操作でポイントtoポイントモーションを作成・編集するツール

使用方法:
    python motion_editor.py --robot.type=so100_follower --robot.port=/dev/ttyUSB0

キーボードコマンド:
    WASD: 各軸の移動 (W/S: shoulder_pan, A/D: shoulder_lift)
    IJKL: その他の軸 (I/K: elbow_flex, J/L: wrist_flex)
    Q/E: wrist_roll
    Z/X: gripper
    M: 現在位置をポイントとして記録
    P: 記録されたモーションを再生
    S: モーションをファイルに保存
    L: モーションをファイルから読み込み
    R: モーションをリセット（2回押しで確定）
    ESC: 終了
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List

import draccus
from lerobot.robots import (
    RobotConfig,
    make_robot_from_config,
    so101_follower,  # noqa: F401
)
from lerobot.utils.utils import init_logging

from motion_utils import (
    Motion,
    MotionPoint,
    apply_pid_settings,
    create_motion_filename,
    get_home_position,
    load_motion_from_file,
    save_motion_to_file,
)

# pynput の import
PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.warning("pynput not available. Keyboard control will be disabled.")


@dataclass
class MotionEditorConfig:
    robot: RobotConfig
    # キーボード操作の移動量（度またはパーセント）
    step_size: float = 10.0
    # 動作の実行速度（Hz）
    control_frequency: float = 30.0
    # モーションファイルの保存先
    motion_dir: str = "./motions"
    # デフォルトファイル名
    default_motion_file: str = "motion.json"
    # 起動時にホームポジションに移動するか
    go_to_home_on_startup: bool = True
    # ホーム移動の速度（秒）
    home_move_duration: float = 1.0
    # PID制御の最適化を適用するか
    optimize_pid: bool = True


class MotionEditor:
    """ポイントtoポイントのモーション編集ツール"""

    def __init__(self, config: MotionEditorConfig):
        self.config = config
        self.robot = make_robot_from_config(config.robot)

        # モーションデータ
        self.motion_points: List[MotionPoint] = []
        self.current_point_index = 0

        # キーボード制御
        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None

        # 制御用変数
        self.running = True
        self.recording_mode = True

        # ファイル管理 - スクリプトファイルの場所を基準にする
        if config.motion_dir.startswith("./"):
            # 相対パスの場合は、スクリプトファイルの場所を基準にする
            script_dir = Path(__file__).parent
            self.motion_dir = script_dir / config.motion_dir[2:]  # "./" を除去
        else:
            self.motion_dir = Path(config.motion_dir)

        self.motion_dir.mkdir(exist_ok=True)
        print(f"📁 モーション保存先: {self.motion_dir.absolute()}")

        # PID状態管理
        self.pid_enabled = config.optimize_pid

        # ステップサイズ管理
        self.current_step_size = config.step_size

        # ホームポジション（安全な初期位置）
        self.home_position = get_home_position()

        init_logging()

    def _get_key_mapping(self) -> Dict[str, Dict[str, float]]:
        """現在のステップサイズでキーボードと関節の対応を取得"""
        return {
            "w": {"shoulder_pan.pos": self.current_step_size},
            "s": {"shoulder_pan.pos": -self.current_step_size},
            "a": {"shoulder_lift.pos": self.current_step_size},
            "d": {"shoulder_lift.pos": -self.current_step_size},
            "i": {"elbow_flex.pos": self.current_step_size},
            "k": {"elbow_flex.pos": -self.current_step_size},
            "j": {"wrist_flex.pos": self.current_step_size},
            "l": {"wrist_flex.pos": -self.current_step_size},
            "q": {"wrist_roll.pos": self.current_step_size},
            "e": {"wrist_roll.pos": -self.current_step_size},
            "z": {"gripper.pos": self.current_step_size},
            "x": {"gripper.pos": -self.current_step_size},
        }

    def _apply_pid_settings(self, use_optimized=None):
        """各モーターにPID設定を適用"""
        if use_optimized is None:
            use_optimized = self.pid_enabled

        apply_pid_settings(self.robot, use_optimized=use_optimized)

    def connect(self):
        """ロボットとキーボードリスナーに接続"""
        print("Connecting to robot...")
        self.robot.connect()

        # PID設定を適用
        self._apply_pid_settings()

        # ホームポジションに移動
        if self.config.go_to_home_on_startup:
            self._go_to_home()

        if PYNPUT_AVAILABLE:
            print("Starting keyboard listener...")
            self.listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
            self.listener.start()
        else:
            print("Warning: pynput not available. Keyboard control disabled.")

    def disconnect(self):
        """接続を切断"""
        if self.listener:
            self.listener.stop()
        if self.robot.is_connected:
            self.robot.disconnect()

    def _go_to_home(self):
        """ロボットをホームポジションに移動"""
        try:
            print("Moving to home position...")
            print("ホームポジション:", self.home_position)

            # 安全のため、現在位置を表示
            current_obs = self.robot.get_observation()
            print("現在位置:")
            for joint in self.home_position.keys():
                current_pos = current_obs.get(joint, 0.0)
                print(f"  {joint}: {current_pos:.2f}")

            # ホームポジションに移動
            self.robot.send_action(self.home_position)

            print(f"Moving to home position... ({self.config.home_move_duration}s)")
            time.sleep(self.config.home_move_duration)
            print("✅ Home position reached!")

        except Exception as e:
            print(f"❌ Error moving to home position: {e}")
            print("Please manually move the robot to a safe position.")

    def _on_key_press(self, key):
        """キー押下イベント処理"""
        try:
            char = key.char if hasattr(key, "char") else str(key)
            self.event_queue.put((char, True))
        except AttributeError:
            # 特殊キー
            self.event_queue.put((str(key), True))

    def _on_key_release(self, key):
        """キー離しイベント処理"""
        try:
            char = key.char if hasattr(key, "char") else str(key)
            self.event_queue.put((char, False))
        except AttributeError:
            self.event_queue.put((str(key), False))

        # ESCで終了
        if key == keyboard.Key.esc or str(key) == "Key.esc":
            self.running = False

    def _process_keyboard_input(self):
        """キーボード入力を処理"""
        # イベントキューから全てのイベントを処理
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()

            if is_pressed:  # キーが押された時のみ処理
                print(f"🔍 キー検出: '{key_char}' (押下)")
                if key_char == "m":  # Mキー - ポイント記録
                    print("🚀 Mキー認識 - 記録開始")
                    self._record_current_position()
                elif key_char == "p":
                    self._play_motion()
                elif key_char.lower() == "c":  # Ctrl+S の代わりに 'c' で保存
                    self._save_motion()
                elif key_char.lower() == "o":  # Ctrl+L の代わりに 'o' で読み込み
                    self._load_motion()
                elif key_char == "r":
                    self._reset_motion()
                elif key_char == "h":
                    self._go_to_home()
                elif key_char == "t":  # Toggle PID
                    self._toggle_pid()
                elif key_char == "f":  # Show help
                    self._print_help()
                elif key_char == "1":  # テスト用
                    print("📊 デバッグ情報:")
                    print(f"  - モーションポイント数: {len(self.motion_points)}")
                    print(f"  - ロボット接続状態: {self.robot.is_connected}")
                    print(f"  - action_features: {list(self.robot.action_features.keys())}")
                elif key_char == "[":  # Decrease step size
                    self._adjust_step_size(-1.0)
                elif key_char == "]":  # Increase step size
                    self._adjust_step_size(1.0)
                elif key_char in self._get_key_mapping():
                    # 関節制御
                    self._move_joint(key_char)

    def _move_joint(self, key: str):
        """指定されたキーに対応する関節を動かす"""
        key_mapping = self._get_key_mapping()
        if key not in key_mapping:
            return

        try:
            # 現在の位置を取得
            current_obs = self.robot.get_observation()

            # デバッグテストと同じ方法：絶対位置指定
            joint_changes = key_mapping[key]
            print(f"Key: {key}, Changes: {joint_changes}")

            for joint_name, delta in joint_changes.items():
                if joint_name in current_obs:
                    old_pos = current_obs.get(joint_name, 0.0)
                    new_pos = old_pos + delta
                    print(f"  {joint_name}: {old_pos:.2f} -> {new_pos:.2f} (delta: {delta})")

                    # シンプルに送信
                    result = self.robot.send_action({joint_name: new_pos})
                    print(f"返答: {result}")

            # 少しだけ待機
            time.sleep(0.1)

            # 動作後の実際の位置を確認
            updated_obs = self.robot.get_observation()
            for joint_name, delta in joint_changes.items():
                if joint_name in updated_obs:
                    actual_pos = updated_obs.get(joint_name, 0.0)
                    movement = actual_pos - current_obs.get(joint_name, 0.0)
                    print(f"  Result: {joint_name} = {actual_pos:.2f}° (moved: {movement:.2f}°)")

        except Exception as e:
            print(f"Error moving joint: {e}")

    def _record_current_position(self):
        """現在の位置をモーションポイントとして記録"""
        print("🎯 スペースキーが押されました - 現在位置を記録中...")
        try:
            current_obs = self.robot.get_observation()
            print(f"📊 ロボット観測データ取得: {len(current_obs)} items")

            # 関節位置のみ抽出
            positions = {}
            for joint_name in self.robot.action_features.keys():
                positions[joint_name] = current_obs.get(joint_name, 0.0)

            print(f"📍 抽出した関節位置: {len(positions)} joints")

            point = MotionPoint(name=f"Point_{len(self.motion_points) + 1}", positions=positions, timestamp=time.time())

            self.motion_points.append(point)
            print(f"✅ Recorded point {len(self.motion_points)}: {point.name}")
            self._print_current_position(positions)

        except Exception as e:
            print(f"❌ Error recording position: {e}")
            import traceback

            traceback.print_exc()

    def _print_current_position(self, positions: Dict[str, float]):
        """現在の位置を表示"""
        print("Current positions:")
        for joint, pos in positions.items():
            print(f"  {joint}: {pos:.2f}")
        print()

    def _play_motion(self):
        """記録されたモーションを再生"""
        if not self.motion_points:
            print("No motion points recorded!")
            return

        print(f"Playing motion with {len(self.motion_points)} points...")

        for i, point in enumerate(self.motion_points):
            print(f"Moving to point {i + 1}: {point.name}")

            try:
                self.robot.send_action(point.positions)
                time.sleep(point.duration)
            except Exception as e:
                print(f"Error during playback at point {i + 1}: {e}")
                break

        print("Motion playback completed!")

    def _save_motion(self):
        """モーションをファイルに保存"""
        if not self.motion_points:
            print("No motion points to save!")
            return

        # 自動ファイル名生成
        filename = create_motion_filename()
        print(f"📁 自動生成ファイル名: {filename}")

        filepath = self.motion_dir / filename

        motion = Motion(
            name=filename.replace(".json", ""),
            points=self.motion_points,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            robot_type=self.robot.name,
        )

        try:
            save_motion_to_file(motion, str(filepath))
            print(f"Motion saved to: {filepath}")
            print(f"Saved {len(self.motion_points)} points")

        except Exception as e:
            print(f"Error saving motion: {e}")

    def _load_motion(self):
        """モーションをファイルから読み込み"""
        # キーボードバッファをクリア
        self._clear_keyboard_buffer()

        filename = input(f"Enter filename to load (default: {self.config.default_motion_file}): ").strip()
        if not filename:
            filename = self.config.default_motion_file

        if not filename.endswith(".json"):
            filename += ".json"

        filepath = self.motion_dir / filename

        if not filepath.exists():
            print(f"File not found: {filepath}")
            return

        try:
            motion = load_motion_from_file(str(filepath))
            self.motion_points = motion.points

            print(f"Motion loaded from: {filepath}")
            print(f"Loaded {len(self.motion_points)} points")
            print(f"Robot type: {motion.robot_type}")

        except Exception as e:
            print(f"Error loading motion: {e}")

    def _toggle_pid(self):
        """PID設定をオン/オフ切り替え"""
        self.pid_enabled = not self.pid_enabled
        status = "最適化PID" if self.pid_enabled else "デフォルトPID"
        print(f"\n🔄 PID設定を切り替えました: {status}")

        # 即座にPID設定を適用
        self._apply_pid_settings(self.pid_enabled)

        print(f"現在のPID状態: {status}")
        if self.pid_enabled:
            print("  - 全モーター: P=20, I=5, D=32 (最適化PID)")
        else:
            print("  - 全モーター: P=16, I=0, D=32 (LeRobotデフォルト)")
        print("Tキーで再度切り替えできます\n")

    def _adjust_step_size(self, delta: float):
        """ステップサイズを調整"""
        old_step = self.current_step_size
        self.current_step_size = max(1.0, self.current_step_size + delta)  # 最小1度
        print(f"\n📏 ステップサイズ変更: {old_step:.1f}° → {self.current_step_size:.1f}°")
        print(f"現在のステップサイズ: {self.current_step_size:.1f}°")
        print("[ ] キーでステップサイズ調整できます\n")

    def _clear_keyboard_buffer(self):
        """キーボードバッファをクリア"""
        print("\n[キーボード入力をクリア中...]")

        # キーボードリスナーを一時停止
        if hasattr(self, "listener") and self.listener:
            self.listener.stop()
            time.sleep(0.1)

        # イベントキューを完全にクリア
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except Exception:
                break

        # 追加のクリア処理
        time.sleep(0.5)
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except Exception:
                break

        # キーボードリスナーを再開
        if PYNPUT_AVAILABLE:
            self.listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
            self.listener.start()
            time.sleep(0.1)

        print("[バッファクリア完了 - キーボードリスナー再開]")

    def _reset_motion(self):
        """記録されたモーションをリセット"""
        if self.motion_points:
            # 確認なしで即座にリセット（安全のため2回押しが必要）
            if not hasattr(self, "_reset_confirm_pending"):
                self._reset_confirm_pending = True
                print(f"⚠️  {len(self.motion_points)}個のポイントをリセットします")
                print("🔄 もう一度Rキーを押して確定してください")
            else:
                # 2回目のRキー押下でリセット実行
                delattr(self, "_reset_confirm_pending")
                self.motion_points.clear()
                print("✅ Motion reset!")
        else:
            print("No motion points to reset.")

    def _print_help(self):
        """ヘルプを表示"""
        pid_status = "試験PID" if self.pid_enabled else "デフォルトPID"
        help_text = f"""
Motion Editor - Keyboard Controls:
==================================

Joint Control:
  W/S: shoulder_pan +/-
  A/D: shoulder_lift +/-
  I/K: elbow_flex +/-
  J/L: wrist_flex +/-
  Q/E: wrist_roll +/-
  Z/X: gripper +/-

Motion Commands:
  M: Record current position as point
  P: Play recorded motion
  C: Save motion to file
  O: Load motion from file
  R: Reset motion (press twice to confirm)
  H: Go to home position
  T: Toggle PID settings (Default ⟷ Test)
  [: Decrease step size (-1°)
  ]: Increase step size (+1°)
  F: Show this help menu
  ESC: Exit

Current Status:
  Points recorded: {len(self.motion_points)}
  Robot: {self.robot.name}
  Connected: {self.robot.is_connected}
  PID: {pid_status}
  Step Size: {self.current_step_size:.1f}°
"""
        print(help_text)

    def run(self):
        """メインループを実行"""
        try:
            self.connect()

            print("\n" + "=" * 50)
            print("Motion Editor Started")
            print("=" * 50)
            self._print_help()

            # メインループ
            while self.running:
                try:
                    if PYNPUT_AVAILABLE:
                        self._process_keyboard_input()
                    else:
                        # キーボード制御が利用できない場合の簡易コマンド
                        cmd = input("Command (h for help, q to quit): ").lower()
                        if cmd == "q":
                            break
                        elif cmd == "h" or cmd == "f":
                            self._print_help()
                        elif cmd == "p":
                            self._play_motion()
                        elif cmd == "s":
                            self._save_motion()
                        elif cmd == "l":
                            self._load_motion()
                        elif cmd == "r":
                            self._reset_motion()
                        elif cmd == "t":
                            self._toggle_pid()
                        elif cmd == "[":
                            self._adjust_step_size(-1.0)
                        elif cmd == "]":
                            self._adjust_step_size(1.0)

                    time.sleep(1.0 / self.config.control_frequency)

                except KeyboardInterrupt:
                    break

        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            print("Shutting down...")
            self.disconnect()


@draccus.wrap()
def main(cfg: MotionEditorConfig):
    """メイン関数"""
    editor = MotionEditor(cfg)
    editor.run()


if __name__ == "__main__":
    main(draccus.parse(MotionEditorConfig))
