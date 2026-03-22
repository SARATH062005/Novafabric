import time
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

def main():
    config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        id="my_leader_arm",   # this is your current follower calibration id
        use_degrees=True,
        disable_torque_on_disconnect=True,
    )

    robot = SOFollower(config)

    robot.connect()
    try:
        print("Follower connected")

        obs = robot.get_observation()
        print("Current observation:")
        print(obs)

        action = {
            "shoulder_pan.pos": obs["shoulder_pan.pos"] + 5.0,
            "shoulder_lift.pos": obs["shoulder_lift.pos"],
            "elbow_flex.pos": obs["elbow_flex.pos"],
            "wrist_flex.pos": obs["wrist_flex.pos"],
            "wrist_roll.pos": obs["wrist_roll.pos"],
            "gripper.pos": obs["gripper.pos"],
        }

        print("Sending action...")
        robot.send_action(action)
        time.sleep(1.0)

        print("Returning...")
        robot.send_action(obs)
        time.sleep(1.0)

    finally:
        if robot.is_connected:
            robot.disconnect()
            print("Follower disconnected")

if __name__ == "__main__":
    main()