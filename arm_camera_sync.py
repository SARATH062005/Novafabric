import time
import cv2
from copy import deepcopy

from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def main():
    # 1. Initialize Robot Arm
    print("Initializing Robot Arm...")
    config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",              # follower port
        id="my_leader_arm",              # your current follower calibration id
        use_degrees=True,
        disable_torque_on_disconnect=True,
    )

    robot = SOFollower(config)

    try:
        robot.connect()
        print("Follower connected")
    except Exception as e:
        print(f"Failed to connect to the robot arm: {e}")
        return

    # Read current pose and use it as home
    home_pose = robot.get_observation()
    
    # Pre-calculate an extended pose for demonstration
    extend_pose = deepcopy(home_pose)
    extend_pose["shoulder_lift.pos"] = clamp(home_pose["shoulder_lift.pos"] + 10.0, -120.0, 120.0)
    extend_pose["elbow_flex.pos"] = clamp(home_pose["elbow_flex.pos"] - 20.0, -120.0, 120.0)
    extend_pose["wrist_flex.pos"] = clamp(home_pose["wrist_flex.pos"] + 10.0, -120.0, 120.0)

    # 2. Initialize Camera
    print("Initializing Camera...")
    # Based on camera.py, camera index is set to 2
    camera_index = 2
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Cannot open USB camera at index {camera_index}")
        robot.disconnect()
        return

    print("--- Arm & Camera Integration Started ---")
    print("Press 'q' to quit, 'e' to extend arm, 'h' to return home")

    try:
        while True:
            # Read camera frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read camera frame")
                break

            # Read arm observation
            obs = robot.get_observation()
            
            # Extract basic joint positions to display on the video frame
            shoulder_pan = obs.get("shoulder_pan.pos", 0.0)
            shoulder_lift = obs.get("shoulder_lift.pos", 0.0)
            elbow_flex = obs.get("elbow_flex.pos", 0.0)
            
            # Overlay arm metrics and keyboard instructions on the video feed
            text1 = f"Pan: {shoulder_pan:.1f} | Lift: {shoulder_lift:.1f} | Elbow: {elbow_flex:.1f}"
            cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "[h]: Home | [e]: Extend | [q]: Quit", (10, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show the video UI
            cv2.imshow("Robotic Arm & Camera", frame)

            # Handle key events with 1ms delay
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('e'):
                print("Moving to extended pose...")
                robot.send_action(extend_pose)
            elif key == ord('h'):
                print("Returning to home pose...")
                robot.send_action(home_pose)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        # 3. Graceful Cleanup
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Follower disconnected and camera released.")

if __name__ == "__main__":
    main()
