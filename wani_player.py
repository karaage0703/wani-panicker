#!/usr/bin/env python3

"""
Wani Player - キーボード0-5でワニモーション00-05を再生するツール

使用方法:
    python wani_player.py --robot.type=so101_follower --robot.id=lerobot_follower --robot.port=/dev/ttyUSB0

キーボードコマンド:
    0: motion_wani_00.json を再生
    1: motion_wani_01.json を再生
    2: motion_wani_02.json を再生
    3: motion_wani_03.json を再生
    4: motion_wani_04.json を再生
    5: motion_wani_05.json を再生
    i: motion_wani_00_02.json を再生（中間ポーズ）
    h: ホームポジションに移動
    ESC: 終了
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import draccus
from lerobot.robots import (
    RobotConfig,
    make_robot_from_config,
    so101_follower,  # noqa: F401
)
from lerobot.utils.utils import init_logging

from motion_utils import apply_pid_settings, get_home_position, load_motion_from_file

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
class WaniPlayerConfig:
    robot: RobotConfig
    # モーションファイルディレクトリ
    motion_dir: str = "./motions"
    # 再生速度の倍率 (1.0が通常速度)
    speed: float = 0.5  # デフォルトを0.5倍速（安全のため）
    # デバッグ情報を表示するか
    verbose: bool = False
    # 開始時にホームポジションに移動するか
    go_to_home: bool = True
    # ホーム移動の時間（秒）
    home_duration: float = 3.0
    # モーション終了後にホームポジションに戻るか
    return_to_home: bool = False
    # モーション終了後の待機時間（秒）
    end_hold_time: float = 1.0
    # 最適化PID設定を適用するか
    use_optimized_pid: bool = True


class WaniPlayer:
    """ワニモーション再生クラス"""

    def __init__(self, cfg: WaniPlayerConfig):
        self.cfg = cfg
        self.robot = None
        self.should_exit = False
        self.motions: Dict[str, object] = {}  # モーションキャッシュ
        self.is_playing_motion = False  # モーション再生中フラグ

        # ログレベル設定
        if cfg.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

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

        # ホームポジションに移動
        if cfg.go_to_home:
            self._go_to_home()

        # モーションファイルを事前に読み込み
        self._load_all_motions()

    def _go_to_home(self):
        """ロボットをホームポジションに移動"""
        print("🏠 ホームポジションに移動中...")
        home_position = get_home_position()

        try:
            # 現在位置を表示
            if self.cfg.verbose:
                current_obs = self.robot.get_observation()
                print("現在位置:")
                for joint in home_position.keys():
                    current_pos = current_obs.get(joint, 0.0)
                    print(f"  {joint}: {current_pos:.2f}")

            # ホームポジションに移動
            self.robot.send_action(home_position)
            print(f"⏱️  ホーム移動時間: {self.cfg.home_duration}秒")
            time.sleep(self.cfg.home_duration)
            print("✅ ホームポジション到達")

        except Exception as e:
            print(f"❌ ホーム移動エラー: {e}")

    def _load_all_motions(self):
        """全てのワニモーションファイルを事前に読み込み"""
        print("📚 ワニモーションファイルを読み込み中...")

        # 中間ポーズファイルを読み込み
        intermediate_file = "motion_wani_00_02.json"
        intermediate_path = Path(self.cfg.motion_dir) / intermediate_file
        if intermediate_path.exists():
            try:
                intermediate_motion = load_motion_from_file(str(intermediate_path))
                self.motions["i"] = intermediate_motion  # 'i'キーで中間ポーズ
                print(f"✅ {intermediate_file}: {intermediate_motion.name} (中間ポーズ)")
            except Exception as e:
                print(f"❌ {intermediate_file} 読み込みエラー: {e}")
        else:
            print(f"⚠️  {intermediate_file} が見つかりません（中間ポーズ用）")

        for i in range(6):  # 0-5
            motion_file = f"motion_wani_{i:02d}.json"
            motion_path = Path(self.cfg.motion_dir) / motion_file

            if motion_path.exists():
                try:
                    motion = load_motion_from_file(str(motion_path))
                    self.motions[str(i)] = motion
                    print(f"✅ {motion_file}: {motion.name}")
                except Exception as e:
                    print(f"❌ {motion_file} 読み込みエラー: {e}")
            else:
                print(f"⚠️  {motion_file} が見つかりません")

        print(f"📖 読み込み完了: {len(self.motions)}個のモーション")

    def _on_key_press(self, key):
        """キー押下イベント処理"""
        try:
            if key == keyboard.Key.esc:
                print("\n🔚 ESCキーが押されました - 終了します")
                self.should_exit = True
                return False  # リスナーを停止

            # 文字キーの処理
            if hasattr(key, "char") and key.char:
                if key.char in "012345":
                    motion_number = key.char
                    if motion_number in self.motions:
                        print(f"\n🎯 モーション{motion_number}を再生します")
                        self._play_motion(self.motions[motion_number])
                    else:
                        print(f"\n❌ モーション{motion_number}が読み込まれていません")

                elif key.char.lower() == "h":
                    print("\n🏠 ホームポジションに移動します")
                    self._go_to_home()

                elif key.char.lower() == "i":
                    if "i" in self.motions:
                        print("\n🔄 中間ポーズに移動します")
                        self._play_motion(self.motions["i"])
                    else:
                        print("\n❌ 中間ポーズファイルが読み込まれていません")

        except AttributeError:
            pass

    def _play_motion(self, motion):
        """モーションを再生"""
        if not motion.points:
            print("❌ モーションポイントがありません")
            return False

        self.is_playing_motion = True  # 再生開始
        print(f"▶️  モーション再生開始: {motion.name}")
        print(f"   速度倍率: {self.cfg.speed}x")

        try:
            for i, point in enumerate(motion.points, 1):
                print(f"📍 ポイント {i}/{len(motion.points)}: {point.name}")

                # デバッグ情報
                if self.cfg.verbose:
                    for joint, pos in point.positions.items():
                        print(f"     {joint}: {pos:.2f}")

                # ロボットを目標位置に移動
                self.robot.send_action(point.positions)

                # 移動時間待機
                time.sleep(point.duration / self.cfg.speed)

                # 移動完了を待機
                time.sleep(0.1)

            print("✅ モーション再生完了")

            # モーション終了後の処理
            if self.cfg.return_to_home:
                print(f"⏱️  終了位置で{self.cfg.end_hold_time}秒待機...")
                time.sleep(self.cfg.end_hold_time)
                print("🏠 ホームポジションに戻り中...")
                self._go_to_home()

            return True

        except Exception as e:
            print(f"❌ モーション再生エラー: {e}")
            return False
        finally:
            self.is_playing_motion = False  # 再生終了

    def run(self):
        """メイン実行"""
        if not PYNPUT_AVAILABLE:
            print("❌ pynputが利用できないため、キーボード操作ができません")
            return

        print("\n🎮 ワニプレイヤー起動完了!")
        print("=== キーボード操作 ===")
        print("0-5: ワニモーション00-05を再生")
        print("i: 中間ポーズに移動 (motion_wani_00_02.json)")
        print("h: ホームポジションに移動")
        print("ESC: 終了")
        print("========================")

        # キーボードリスナーを開始
        listener = keyboard.Listener(on_press=self._on_key_press)
        listener.start()

        try:
            # メインループ - キー入力を待機
            while not self.should_exit:
                # モーション再生中でない場合のみ現在位置を保持
                if not self.is_playing_motion:
                    current_obs = self.robot.get_observation()
                    self.robot.send_action(current_obs)
                time.sleep(0.5)  # 0.5秒間隔で保持

        except KeyboardInterrupt:
            print("\n🔚 Ctrl+Cで終了します")
        finally:
            listener.stop()
            # ロボット切断
            if self.robot and hasattr(self.robot, "is_connected") and self.robot.is_connected:
                try:
                    self.robot.disconnect()
                    print("🔌 ロボットから切断しました")
                except Exception as e:
                    print(f"⚠️ ロボット切断時にエラー: {e}")


@draccus.wrap()
def main(cfg: WaniPlayerConfig):
    """メイン関数"""
    init_logging()

    # モーションディレクトリ作成
    Path(cfg.motion_dir).mkdir(exist_ok=True)

    player = WaniPlayer(cfg)
    player.run()


if __name__ == "__main__":
    main()
