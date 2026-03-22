import time
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

def main():
    config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        id="my_leader_arm",
        use_degrees=True,
        disable_torque_on_disconnect=True,
    )

    robot = SOFollower(config)

    robot.connect()
    try:
        obs = robot.get_observation()
        print("Obs:", obs)

        action = dict(obs)
        action["shoulder_pan.pos"] = obs["shoulder_pan.pos"] + 5.0

        robot.send_action(action)
        time.sleep(1.0)

        robot.send_action(obs)
        time.sleep(1.0)

    finally:
        if robot.is_connected:
            robot.disconnect()

if __name__ == "__main__":
    main()