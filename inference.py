import argparse
import time
import cv2
import numpy as np
import torch
from pathlib import Path

from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to the trained model checkpoint (e.g. outputs/train/act_so100/checkpoints/005000/pretrained_model)")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset used for training (to load statistics)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--usb-camera-index", type=int, default=2, help="USB Camera index")
    parser.add_argument("--webcam-index", type=int, default=0, help="Laptop webcam index")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()

    # 1. Load Dataset Metadata (for statistics and feature info)
    print(f"Loading metadata from {args.dataset_path}...")
    # In LeRobot v3.0, if root is provided, it must be the path to the repository directory
    ds_meta = LeRobotDatasetMetadata(Path(args.dataset_path).name, root=args.dataset_path)

    # 2. Load Policy
    print(f"Loading policy from {args.checkpoint_path}...")
    # Load the config from the checkpoint
    from lerobot.configs.policies import PreTrainedConfig
    policy_cfg = PreTrainedConfig.from_pretrained(args.checkpoint_path)
    policy_cfg.device = args.device
    policy_cfg.pretrained_path = Path(args.checkpoint_path)

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()

    # 3. Create Pre/Post Processors
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg, pretrained_path=args.checkpoint_path)

    # 4. Initialize Robot
    print("Initializing Robot Arm...")
    robot_config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        id="my_leader_arm3",
        use_degrees=True,
        calibration_dir=Path("."),
    )
    robot = SOFollower(robot_config)
    robot.connect(calibrate=False)
    # Enable torque for autonomous execution
    robot.bus.enable_torque()

    # 5. Camera Setup (USB)
    cap_usb = cv2.VideoCapture(args.usb_camera_index)
    cap_usb.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap_usb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_usb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 6. Camera Setup (Webcam)
    cap_webcam = cv2.VideoCapture(args.webcam_index)
    cap_webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap_usb.isOpened():
        print(f"Error: Cannot open USB camera {args.usb_camera_index}")
        return
    if not cap_webcam.isOpened():
        print(f"Warning: Cannot open laptop webcam {args.webcam_index}")

    print("\n--- Autonomous Inference Started ---")
    print("Robot is now controlled by the policy. Press [ESC] to stop.")

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    try:
        while True:
            start_time = time.time()
            
            ret_usb, frame_usb = cap_usb.read()
            ret_web, frame_webcam = cap_webcam.read()
            
            if not ret_usb:
                break
            
            if not ret_web:
                frame_webcam = np.zeros_like(frame_usb)

            # Get current robot state
            obs_robot = robot.get_observation()
            state = np.array([obs_robot.get(f"{name}.pos", 0.0) for name in joint_names], dtype=np.float32)

            # Prepare observation batch for policy
            img_usb_tensor = torch.from_numpy(frame_usb).float() / 255.0
            img_usb_tensor = img_usb_tensor.permute(2, 0, 1).unsqueeze(0).to(args.device)
            
            img_web_tensor = torch.from_numpy(frame_webcam).float() / 255.0
            img_web_tensor = img_web_tensor.permute(2, 0, 1).unsqueeze(0).to(args.device)
            
            obs_dict = {
                "observation.state": torch.from_numpy(state).unsqueeze(0).to(args.device), 
                "observation.images.laptop": img_usb_tensor,
                "observation.images.webcam": img_web_tensor,
            }

            # Preprocess
            obs_dict = preprocessor(obs_dict)
            
            # Predict action
            with torch.inference_mode():
                action = policy.select_action(obs_dict)
            
            # Postprocess
            action = postprocessor(action)
            
            # Convert action to numpy and send to robot
            # Action shape is (batch, action_dim)
            action_np = action.squeeze(0).cpu().numpy()
            
            # Convert to dictionary expected by SOFollower
            action_dict = {f"{name}.pos": val for name, val in zip(joint_names, action_np)}
            
            # Send action to robot
            robot.send_action(action_dict)

            # Visualization
            h, w, _ = frame_usb.shape
            frame_webcam_resized = cv2.resize(frame_webcam, (w, h))
            combined_view = np.hstack((frame_usb, frame_webcam_resized))
            cv2.imshow("Inference (USB | Webcam)", combined_view)
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break

            # Maintain FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0 / args.fps) - elapsed)
            time.sleep(sleep_time)

    finally:
        cap_usb.release()
        cap_webcam.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Inference stopped.")

if __name__ == "__main__":
    main()
