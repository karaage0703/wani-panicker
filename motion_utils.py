#!/usr/bin/env python3

"""
Motion Utils - モーション関連の共通クラスとユーティリティ
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


@dataclass
class MotionPoint:
    """モーションの1つのポイント"""

    name: str  # ポイントの名前
    positions: Dict[str, float]  # 各関節の位置
    duration: float = 1.0  # このポイントまでの移動時間（秒）
    timestamp: float = 0.0  # 記録時のタイムスタンプ


@dataclass
class Motion:
    """モーション全体を表すクラス"""

    name: str
    points: List[MotionPoint]
    created_at: str
    robot_type: str


def save_motion_to_file(motion: Motion, filepath: str) -> None:
    """モーションをファイルに保存"""
    motion_dict = asdict(motion)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(motion_dict, f, indent=2, ensure_ascii=False)


def load_motion_from_file(filepath: str) -> Motion:
    """ファイルからモーションを読み込み"""
    with open(filepath, "r", encoding="utf-8") as f:
        motion_dict = json.load(f)

    # MotionPointオブジェクトに変換
    points = [MotionPoint(**point) for point in motion_dict["points"]]

    return Motion(
        name=motion_dict["name"], points=points, created_at=motion_dict["created_at"], robot_type=motion_dict["robot_type"]
    )


def list_motion_files(motion_dir: str) -> List[str]:
    """motionディレクトリ内のJSONファイル一覧を取得"""
    motion_path = Path(motion_dir)
    if not motion_path.exists():
        return []

    json_files = list(motion_path.glob("*.json"))
    return sorted([f.name for f in json_files])


def create_motion_filename() -> str:
    """タイムスタンプベースのモーションファイル名を生成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"motion_{timestamp}.json"


def get_home_position() -> Dict[str, float]:
    """SO101ロボット用の安全なホームポジションを取得"""
    return {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.0,
        "elbow_flex.pos": 0.0,
        "wrist_flex.pos": 0.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 0.0,
    }


def apply_pid_settings(robot, use_optimized: bool = True) -> bool:
    """ロボットにPID設定を適用

    Args:
        robot: ロボットオブジェクト
        use_optimized: Trueで最適化PID、FalseでデフォルトPID

    Returns:
        bool: 設定成功時True
    """
    import time

    if not use_optimized:
        print("LeRobotデフォルトPID設定を適用中...")
        pid_to_apply = {"p": 16, "i": 0, "d": 32}  # LeRobotデフォルト
    else:
        print("最適化PID設定を適用中...")
        pid_to_apply = {"p": 20, "i": 5, "d": 32}  # 負方向制御改善を狙った最適化設定

    print("モーターごとのPIDゲインを設定中...")

    try:
        # LeRobotと同じ方式：全モーター同じPID設定
        if hasattr(robot, "bus") and hasattr(robot.bus, "motors"):
            print(f"全モーターに適用: P={pid_to_apply['p']}, I={pid_to_apply['i']}, D={pid_to_apply['d']}")
            for motor_name in robot.bus.motors:
                # LeRobotと同じ書き込み方式
                robot.bus.write("P_Coefficient", motor_name, pid_to_apply["p"])
                robot.bus.write("I_Coefficient", motor_name, pid_to_apply["i"])
                robot.bus.write("D_Coefficient", motor_name, pid_to_apply["d"])

                time.sleep(0.01)  # 各設定間で少し待機

        print("✅ PID設定完了")
        return True

    except Exception as e:
        print(f"❌ PID設定エラー: {e}")
        print("デフォルトのPID設定で動作します")
        return False
