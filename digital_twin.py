import pybullet as p
import pybullet_data
import time
import numpy as np
import json
import argparse
import os

from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

def main():
    parser = argparse.ArgumentParser(description="Digital twin integration for SO-100 arm.")
    parser.add_argument("--mode", type=str, choices=["sim_only", "sim_to_real", "real_to_sim"], default="sim_only",
                        help="Mode: sim_only (test manual movement in sim), sim_to_real (control real with sim sliders), real_to_sim (mirror real in sim).")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Port for the real robot.")
    parser.add_argument("--urdf", type=str, default="robot/so_100_arm_5dof.urdf", help="Path to URDF.")
    parser.add_argument("--set_home", action="store_true", help="Record current physical position as the home zero-point.")
    
    args = parser.parse_args()

    # 1. Setup Real Robot (skip if sim_only)
    robot = None
    if args.mode != "sim_only":
        print(f"Connecting to real robot on {args.port}...")
        config = SOFollowerRobotConfig(
            port=args.port,
            id="my_leader_arm",
            use_degrees=True,  # Match testing script, needs conversion to radians for sim
            disable_torque_on_disconnect=True,
        )
        
        robot = SOFollower(config)
        robot.connect()
        print("Real robot connected.")
        
        home_file = "home_position.json"
        home_offsets = {}
        
        if args.set_home:
            # Analyze and save current pos as Home
            print("Analyzing current robot position to set as home...")
            time.sleep(1.0) # Wait for steady read
            obs = robot.get_observation()
            for key, val in obs.items():
                if "pos" in key:
                    home_offsets[key] = float(val)
            
            with open(home_file, "w") as f:
                json.dump(home_offsets, f, indent=4)
            print(f"Saved current position as home in {home_file}: {home_offsets}")
        else:
            if os.path.exists(home_file):
                with open(home_file, "r") as f:
                    home_offsets = json.load(f)
                print(f"Loaded home offsets from {home_file}")
            else:
                print("No home offsets loaded. Using raw values. Use --set_home to calibrate.")

    try:
        # 2. Setup Simulation
        print("Setting up simulation...")
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # Start position
        start_pos = [0, 0, 0]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        robot_id = p.loadURDF(args.urdf, start_pos, start_orn, useFixedBase=True)

        # Mapping between PyBullet joint names and lerobot joint names
        joint_mapping = {
            "Shoulder_Rotation": "shoulder_pan.pos",
            "Shoulder_Pitch": "shoulder_lift.pos",
            "Elbow": "elbow_flex.pos",
            "Wrist_Pitch": "wrist_flex.pos",
            "Wrist_Roll": "wrist_roll.pos",
            "Gripper": "gripper.pos"
        }
        
        # Build index
        pb_joints = {}
        sliders = {}
        gripper_link_index = None

        for j in range(p.getNumJoints(robot_id)):
            joint_info = p.getJointInfo(robot_id, j)
            joint_name = joint_info[1].decode("utf-8")
            joint_type = joint_info[2]
            
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                lower = joint_info[8]
                upper = joint_info[9]
                if lower < -np.pi: lower = -np.pi
                if upper > np.pi: upper = np.pi
                
                pb_joints[joint_name] = j
                
                # Create slider for testing if not just mirroring
                if args.mode in ["sim_only", "sim_to_real"]:
                    slider = p.addUserDebugParameter(joint_name, lower, upper, 0)
                    sliders[joint_name] = slider

            if joint_info[12].decode("utf-8") == "Moving_Jaw":
                gripper_link_index = j

        print(f"Running in {args.mode} mode.")
        print("Press Ctrl+C to stop.")

        # 3. Main Loop
        while True:
            if args.mode == "sim_only":
                # Only run sim with manual sliders
                for pb_name in pb_joints:
                    if pb_name in sliders:
                        target = p.readUserDebugParameter(sliders[pb_name])
                        p.setJointMotorControl2(robot_id, pb_joints[pb_name], p.POSITION_CONTROL, targetPosition=target, force=200)
            
            elif args.mode == "sim_to_real" and robot is not None:
                # Read sliders, apply to pybullet, convert to degrees and send to real robot
                action = {}
                for pb_name, real_name in joint_mapping.items():
                    if pb_name in sliders:
                        target_rad = p.readUserDebugParameter(sliders[pb_name])
                        p.setJointMotorControl2(robot_id, pb_joints[pb_name], p.POSITION_CONTROL, targetPosition=target_rad, force=200)
                        
                        target_deg = target_rad * 180.0 / np.pi
                        if real_name in home_offsets:
                            target_deg += home_offsets[real_name]  # Apply home offset shift
                        
                        action[real_name] = target_deg
                
                # Update observation and send real action
                obs = robot.get_observation()
                full_action = dict(obs)
                full_action.update(action)
                robot.send_action(full_action)
                
            elif args.mode == "real_to_sim" and robot is not None:
                # Fetch degrees from real robot, convert to radians, update Pybullet
                obs = robot.get_observation()
                
                for pb_name, real_name in joint_mapping.items():
                    if pb_name in pb_joints and real_name in obs:
                        target_deg = obs[real_name]
                        if real_name in home_offsets:
                            target_deg -= home_offsets[real_name]  # Remove home offset shift to get zero-based value
                            
                        target_rad = target_deg * np.pi / 180.0
                        p.setJointMotorControl2(robot_id, pb_joints[pb_name], p.POSITION_CONTROL, targetPosition=target_rad, force=200)

            # Optional: update camera view based on gripper
            if gripper_link_index is not None:
                link_state = p.getLinkState(robot_id, gripper_link_index)
                gripper_pos, gripper_orn = link_state[4], link_state[5]
                
                rot_matrix = np.array(p.getMatrixFromQuaternion(gripper_orn)).reshape(3,3)
                forward_vec, up_vec = rot_matrix[:,0], rot_matrix[:,2]
                cam_target = np.array(gripper_pos) + forward_vec * 0.1
                
                view_matrix = p.computeViewMatrix(gripper_pos, cam_target, up_vec)
                # p.getCameraImage(320, 240, viewMatrix=view_matrix) # Rendering might be slow

            p.stepSimulation()
            time.sleep(1./60.) # Update at ~60Hz

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Disconnecting...")
        p.disconnect()
        if robot is not None and robot.is_connected:
            robot.disconnect()

if __name__ == "__main__":
    main()
