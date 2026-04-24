import argparse
import time
import cv2
import numpy as np
from pathlib import Path
from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="sarath/so100_demo", help="Dataset identifier")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to record")
    parser.add_argument("--usb-camera-index", type=int, default=2, help="External USB Camera index")
    parser.add_argument("--webcam-index", type=int, default=0, help="Laptop webcam index")
    args = parser.parse_args()

    # 1. Initialize Robot Arm (Single Arm setup)
    print("Initializing Robot Arm...")
    config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        id="my_leader_arm3", # Matches user's existing setup
        use_degrees=True,
        disable_torque_on_disconnect=True,
        calibration_dir=Path("."),
    )
    robot = SOFollower(config)
    robot.connect(calibrate=False) # Assume already calibrated
    robot.bus.disable_torque() # Manual teaching mode

    # 2. Camera Setup (USB Camera)
    cap_usb = cv2.VideoCapture(args.usb_camera_index)
    cap_usb.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap_usb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_usb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_usb.set(cv2.CAP_PROP_FPS, args.fps)

    # 3. Camera Setup (Laptop Webcam)
    cap_webcam = cv2.VideoCapture(args.webcam_index)
    cap_webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_webcam.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap_usb.isOpened():
        print(f"Error: Cannot open USB camera at index {args.usb_camera_index}")
        return
    if not cap_webcam.isOpened():
        print(f"Warning: Cannot open laptop webcam at index {args.webcam_index}")

    # 3. Create/Load Dataset
    # Define features for SO-100 (6 joints: pan, lift, elbow, wrist_flex, wrist_roll, gripper)
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(joint_names),), "names": joint_names},
        "action": {"dtype": "float32", "shape": (len(joint_names),), "names": joint_names},
        "observation.images.laptop": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
        "observation.images.webcam": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
    }

    dataset_path = Path("datasets") / args.repo_id
    # Check if this is a valid local dataset (meta/tasks.parquet must exist)
    is_valid_local = (dataset_path / "meta" / "tasks.parquet").exists()

    if is_valid_local:
        print(f"Loading existing dataset at {dataset_path}")
        dataset = LeRobotDataset(args.repo_id, root=dataset_path)
    else:
        if dataset_path.exists():
            print(f"Directory {dataset_path} exists but is missing metadata. Removing it to start fresh...")
            import shutil
            shutil.rmtree(dataset_path)
        
        print(f"Creating new dataset at {dataset_path}")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=features,
            robot_type="so100",
            root=dataset_path,
            use_videos=True,
        )

    print("\n--- Recording Instructions ---")
    print("- [SPACE]: Start/Stop recording an episode")
    print("- [ESC]: Exit script")
    print("- MOVE THE ARM MANUALLY to demonstrate the task.\n")

    recording = False
    episode_frames = []

    try:
        while True:
            ret_usb, frame_usb = cap_usb.read()
            ret_web, frame_webcam = cap_webcam.read()
            
            if not ret_usb:
                break
            
            # If webcam fails, use a blank frame or skip
            if not ret_web:
                frame_webcam = np.zeros_like(frame_usb)

            # Display status (Side-by-Side)
            h, w, _ = frame_usb.shape
            # Resize webcam if resolution differs
            frame_webcam_resized = cv2.resize(frame_webcam, (w, h))
            
            combined_frame = np.hstack((frame_usb, frame_webcam_resized))
            display_frame = combined_frame.copy()
            
            status_text = "RECORDING" if recording else "IDLE"
            if recording:
                status_text += f" ({len(episode_frames)} frames)"
            
            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(display_frame, f"STATUS: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(display_frame, f"Episodes: {dataset.num_episodes}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Record Demonstration (USB | Webcam)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC
                print("Exit requested.")
                break
            elif key == ord(' '): # SPACE
                if not recording:
                    print(f"\n--- Starting episode {dataset.num_episodes + 1} ---")
                    recording = True
                    episode_frames = []
                else:
                    if len(episode_frames) > 0:
                        print(f"Episode finished. Saving {len(episode_frames)} frames...")
                        recording = False
                        # Save episode to dataset
                        for f in episode_frames:
                            dataset.add_frame(f)
                        dataset.save_episode()
                        print(f"Episode {dataset.num_episodes} saved successfully!")
                    else:
                        print("Warning: Episode too short, discarding.")
                        recording = False

            if recording:
                # Capture robot state
                obs = robot.get_observation()
                # Extract joint positions in the correct order
                state = np.array([obs.get(f"{name}.pos", 0.0) for name in joint_names], dtype=np.float32)
                
                dataset_frame = {
                    "observation.state": state,
                    "action": state,
                    "observation.images.laptop": frame_usb,
                    "observation.images.webcam": frame_webcam,
                    "task": "demonstration",
                }
                episode_frames.append(dataset_frame)
                
                # Maintain roughly 30 FPS
                time.sleep(1.0 / args.fps)

    except Exception as e:
        print(f"\nAn error occurred during recording: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap_usb.release()
        cap_webcam.release()
        cv2.destroyAllWindows()
        if hasattr(dataset, "finalize"):
            dataset.finalize()
        robot.disconnect()
        print("Robot disconnected and cleanup finished.")

if __name__ == "__main__":
    main()
