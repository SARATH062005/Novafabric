import os
from pathlib import Path
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

def main():
    arm_id = "my_leader_arm3"
    print(f"--- SO-100 Calibration Utility for ID: {arm_id} ---")
    
    # 1. Define calibration config
    config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        id=arm_id,
        use_degrees=True,
        disable_torque_on_disconnect=True,
        calibration_dir=Path("."),
    )

    # 2. Check for existing calibration
    cal_file = Path(f"{arm_id}.json")
    if cal_file.exists():
        print(f"Found existing calibration: {cal_file}")
        # Manual check to ensure user wants to redo it
        print("To reteach calibration, we must remove the current file.")
        
        bak_file = cal_file.with_suffix(".json.bak")
        cal_file.replace(bak_file)
        print(f"Old calibration backed up to {bak_file}")

    # 3. Create robot and connect with calibrate=True
    try:
        robot = SOFollower(config)
        print("\nConnecting to robot... Starting mechanical calibration.")
        print("Follow the terminal instructions to:")
        print("1. Set all joints to their 0-degree markers.")
        print("2. Move joints to their mechanical range limits.")
        robot.connect(calibrate=True)
        print(f"\nCalibration complete! File saved as {cal_file}")
        robot.disconnect()
    except Exception as e:
        print(f"Error during calibration: {e}")

if __name__ == "__main__":
    main()

