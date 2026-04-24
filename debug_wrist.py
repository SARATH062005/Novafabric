import time
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig
from pathlib import Path

def main():
    config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        id="my_leader_arm3",
        use_degrees=True,
    )
    robot = SOFollower(config)
    robot.connect()
    
    print("\n--- MOTOR DEBUGGER ---")
    try:
        while True:
            obs = robot.get_observation()
            print("\nPresent Positions:")
            for k, v in obs.items():
                if k.endswith(".pos"):
                    print(f"  {k:20}: {v:7.2f}")
            
            print("\nTry moving the arm (torque is currently ENABLED by default on connect).")
            print("To check if a motor is RESPONDING, see if the value changes when it moves.")
            print("Wait, I'll DISABLE torque now for 10 seconds so you can manually move it.")
            robot.bus.disable_torque()
            
            for i in range(10, 0, -1):
                obs = robot.get_observation()
                wr = obs.get("wrist_roll.pos", "MISSING")
                print(f"Torque OFF. Move the wrist roll! [{i}s remaining] Wrist Roll: {wr}")
                time.sleep(1)
            
            print("\nRe-enabling torque...")
            robot.bus.enable_torque()
            break
            
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
