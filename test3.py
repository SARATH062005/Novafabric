import time
from copy import deepcopy

from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main():
    config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",              # follower port
        id="my_leader_arm",              # your current follower calibration id
        use_degrees=True,
        disable_torque_on_disconnect=True,
    )

    robot = SOFollower(config)

    robot.connect()
    try:
        print("Follower connected")

        # Read current pose and use it as home
        home = robot.get_observation()
        print("Home observation:")
        print(home)

        # Build a new target pose by extending the arm a little
        # Keep changes small first. Your calibration/assembly is still fresh.
        extend_pose = deepcopy(home)

        # Typical extension idea:
        # - shoulder_lift slightly up/down depending on your build
        # - elbow_flex more open
        # - wrist_flex compensated slightly
        #
        # These signs may need small tuning based on your arm orientation.
        extend_pose["shoulder_lift.pos"] = clamp(home["shoulder_lift.pos"] + 10.0, -120.0, 120.0)
        extend_pose["elbow_flex.pos"] = clamp(home["elbow_flex.pos"] - 20.0, -120.0, 120.0)
        extend_pose["wrist_flex.pos"] = clamp(home["wrist_flex.pos"] + 10.0, -120.0, 120.0)

        print("\nMoving to extended pose...")
        print(extend_pose)
        robot.send_action(extend_pose)
        time.sleep(2.0)

        print("\nReturning to home pose...")
        robot.send_action(home)
        time.sleep(2.0)

        print("\nDone.")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        if robot.is_connected:
            robot.disconnect()
            print("Follower disconnected")


if __name__ == "__main__":
    main()