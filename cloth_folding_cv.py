import time
import cv2
import numpy as np
import json
import os
from pathlib import Path
from copy import deepcopy

# Try to import YOLO for segmentation
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def detect_cloth(frame, model):
    """
    Detects the green kerchief using HSV color segmentation and/or YOLO.
    """
    # 1. HSV Color Segmentation tuned for the Green Kerchief in image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Range for the specific green seen in the screenshot
    # Low H value to catch forest green, high S to ignore brown table
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 180])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours from the color mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    color_detected = False
    cx, cy = None, None

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 5000: # Sufficient size for the kerchief
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                x, y, w, h = cv2.boundingRect(largest)
                
                # Draw the green-specific detection
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 7, (0, 255, 0), -1)
                cv2.putText(frame, "GREEN KERCHIEF DETECTED", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                color_detected = True

    # 2. YOLO Segmentation as a secondary refiner
    if YOLO_AVAILABLE and model is not None:
        results = model.predict(frame, conf=0.35, verbose=False, max_det=1)
        for r in results:
            if r.masks is not None and len(r.masks) > 0:
                # If YOLO finds it, overlay the professional segmentation
                frame[:] = r.plot(boxes=False, labels=True)
                
                # If we didn't get a center from color, get it from YOLO
                if not color_detected and len(r.masks.xy) > 0 and len(r.masks.xy[0]) > 0:
                    mask_pts = r.masks.xy[0] 
                    cx = int(np.mean(mask_pts[:, 0]))
                    cy = int(np.mean(mask_pts[:, 1]))
                    color_detected = True

    return color_detected, (cx, cy)

def main():
    # 1. Initialize YOLO Model
    model = None
    if YOLO_AVAILABLE:
        print("Loading YOLOv11 Segmentation model...")
        # yolo11n-seg.pt is fast and lightweight
        try:
            model = YOLO("yolo11s-seg.pt") 
        except Exception as e:
            print(f"Error loading YOLO weights: {e}")
    else:
        print("Note: 'ultralytics' not found. Using OpenCV fallback for cloth detection.")
        print("Tip: Run 'pip install ultralytics' to enable YOLO segmentation.")

    # 2. Initialize Robot Arm
    print("Initializing Robot Arm...")
    config = SOFollowerRobotConfig(
        port="/dev/ttyACM0",
        id="my_leader_arm3", # Use the calibration currently being worked on
        use_degrees=True,
        disable_torque_on_disconnect=True,
        calibration_dir=Path("."), # Save calibration in THIS folder
    )

    robot = SOFollower(config)

    try:
        # Connect and trigger calibration if the local file is missing
        robot.connect(calibrate=True) 
        print("Robot connected.")
        
        # --- ACCURACY & SPEED CONFIGURATION ---
        fast_accel = 120 
        stiffness = 32 # Increase P_Coefficient for more accurate holding (default 16)
        for motor in robot.bus.motors:
            try:
                robot.bus.write("Acceleration", motor, fast_accel)
                robot.bus.write("P_Coefficient", motor, stiffness)
            except:
                pass
        # INCREASE GRIPPER TORQUE for stronger grip
        if "gripper" in robot.bus.motors:
            try:
                # Max_Torque_Limit is EPROM, usually writeable with torque disabled
                robot.bus.disable_torque("gripper")
                robot.bus.write("Max_Torque_Limit", "gripper", 800, num_retry=3) 
                print("Increased gripper torque limit to 800.")
            except Exception as e:
                print(f"Warning: Could not set gripper torque limit: {e}")

    except Exception as e:
        print(f"Failed to connect to the robot arm: {e}")
        return

    # 3. Camera Setup
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    
    # Optimize camera settings for speed
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Error: Cannot open USB camera at index {camera_index}")
        robot.disconnect()
        return

    # 4. Waypoint Loading
    waypoints_file = "folding_waypoints.json"
    waypoints = []
    if os.path.exists(waypoints_file):
        try:
            with open(waypoints_file, "r") as f:
                waypoints = json.load(f)
            print(f"Loaded {len(waypoints)} waypoints from {waypoints_file}")
        except Exception as e:
            print(f"Error loading waypoints: {e}")

    mode = "TEACHING" 
    playback_index = 0
    last_move_time = 0
    
    # Fast path following speed (seconds per point)
    # Since we record many points continuously, this needs to be very small
    time_between_waypoints = 0.05 
    
    # --- TRAJECTORY VARIABLES ---
    # Since we have high density points, we can reduce interpolation steps to 1 (direct path)
    trajectory_steps = 1 
    
    # --- SMART SAVING VARIABLES ---
    
    # Start in manual teaching mode
    robot.bus.disable_torque()

    print("\n--- Cloth Folding Bot with YOLO Segment ---")
    print("TEACHING MODE (Arm is loose):")
    print("- [ENTER]: Save Waypoint")
    print("- [BACKSPACE]: Delete Last Point")
    print("- [g]: Toggle Gripper (Open/Close)")
    print("- [p]: Finish & Start Playback")
    print("- [c]: Clear All Saved Points")
    print("- [q]: Quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run Cloth Detection (YOLO or CV)
            cloth_visible, target_center = detect_cloth(frame, model)

            # Get latest robot observations for UI
            obs = robot.get_observation()
            
            # --- UI OVERLAYS ---
            # 1. Mode Display
            mode_color = (0, 165, 255) if mode == "TEACHING" else (0, 255, 255)
            cv2.putText(frame, f"MODE: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
            
            # 2. Points saved status
            cv2.putText(frame, f"Points: {len(waypoints)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 3. YOLO Status
            yolo_status = "YOLO ACTIVE" if YOLO_AVAILABLE else "YOLO MISSING (using CV)"
            yolo_color = (0, 255, 0) if YOLO_AVAILABLE else (0, 0, 255)
            cv2.putText(frame, yolo_status, (10, frame.shape[0]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yolo_color, 1)

            # 4. Robot Joints (Fixed Positions for all 6 motors)
            rx = frame.shape[1] - 150 # Right-aligned
            cv2.putText(frame, "JOINTS", (rx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            joints = [
                ("Pan", "shoulder_pan.pos"),
                ("Lift", "shoulder_lift.pos"),
                ("Elbo", "elbow_flex.pos"),
                ("W-Fl", "wrist_flex.pos"),
                ("W-Rl", "wrist_roll.pos"),
                ("Grip", "gripper.pos")
            ]
            for i, (label, key) in enumerate(joints):
                val = obs.get(key, 0.0) or 0.0
                cv2.putText(frame, f"{label}: {val:5.1f}", (rx, 55 + i*22), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # --- LOGIC HANDLING ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if mode == "TEACHING":
                cv2.putText(frame, "[ENTER]: Save | [BACKSPACE]: Undo | Hold Still: Auto Save", 
                            (10, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # --- HIGH-ACCURACY CONTINUOUS TRAJECTORY SAVING ---
                # Save any point that is different enough from the last one (sub-degree accuracy)
                pos_only = {k: v for k, v in obs.items() if isinstance(v, (int, float, str))}
                if cloth_visible:
                    pos_only["ref_cx"], pos_only["ref_cy"] = target_center[0], target_center[1]
                
                # Check distance from the previous waypoint to decide if we should save
                is_new = True
                # Lower threshold (0.2 degrees) for high-precision recording
                move_trigger_dist = 0.2 
                
                if len(waypoints) > 0:
                    last_wp = waypoints[-1]
                    # Calculate cumulative change in motor positions
                    dist = sum([abs(pos_only.get(k, 0) - last_wp.get(k, 0)) for k in pos_only if ".pos" in k])
                    if dist < move_trigger_dist:
                        is_new = False
                
                if is_new:
                    waypoints.append(pos_only)
                    with open(waypoints_file, "w") as f:
                        json.dump(waypoints, f)
                    # print(f"-> Tracepoint {len(waypoints)} recorded.") # too many prints if we keep it on
                
                # Manual Toggle Gripper during teaching
                if key == ord('g'):
                    # Read current gripper pos
                    cur_grip = obs.get("gripper.pos", 0.0)
                    # If it's more than 50, close it; else open it
                    new_grip = 0.0 if cur_grip > 50 else 100.0
                    robot.bus.enable_torque("gripper")
                    robot.send_action({"gripper.pos": new_grip})
                    time.sleep(0.3)
                    robot.bus.disable_torque("gripper")
                    print(f"Gripper toggled to {new_grip}")

                if key == 13 or key == ord('s'): # Enter to save
                    pos_only = {k: v for k, v in obs.items() if isinstance(v, (int, float, str))}
                    
                    # Capture the center of the cloth during training to create a reference
                    if cloth_visible:
                        pos_only["ref_cx"] = target_center[0]
                        pos_only["ref_cy"] = target_center[1]
                    
                    waypoints.append(pos_only)
                    with open(waypoints_file, "w") as f:
                        json.dump(waypoints, f)
                    print(f"-> Waypoint {len(waypoints)} saved.")
                
                elif key == 8 or key == 127: # Backspace key
                    if len(waypoints) > 0:
                        waypoints.pop()
                        with open(waypoints_file, "w") as f:
                            json.dump(waypoints, f)
                        print(f"-> Last waypoint deleted. {len(waypoints)} points remaining.")
                    else:
                        print("-> No waypoints to delete.")
                
                elif key == ord('c'):
                    waypoints = []
                    if os.path.exists(waypoints_file): os.remove(waypoints_file)
                    print("Waypoints cleared.")

                elif key == ord('p'):
                    if len(waypoints) > 0:
                        mode = "PLAYBACK_WAIT"
                        print("Switching to Playback-Wait...")
            
            elif mode == "PLAYBACK_WAIT":
                cv2.putText(frame, "WAITING FOR CLOTH...", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if cloth_visible:
                    print(f"Cloth Detected at {target_center}! Applying global path shift...")
                    
                    # Store a temporary shifted version of the waypoints for this playback session
                    # We deepcopy to avoid corrupting the original waypoints on disk
                    current_path = deepcopy(waypoints)
                    
                    # Calculate and apply offset globally to ALL points
                    first_ref = waypoints[0]
                    if "ref_cx" in first_ref and target_center:
                        trained_cx, trained_cy = first_ref["ref_cx"], first_ref["ref_cy"]
                        current_cx, current_cy = target_center
                        
                        dx = current_cx - trained_cx
                        dy = current_cy - trained_cy
                        
                        # Hardware-tuned gain for 640x480 resolution
                        gain = 0.08 # Adjusted slightly higher for better responsiveness
                        
                        pan_offset = dx * gain
                        lift_offset = -dy * gain # Screen Y increases downwards
                        
                        print(f"Applying Global Offset: Pan {pan_offset:.2f}, Lift {lift_offset:.2f}")
                        
                        for wp in current_path:
                            if "shoulder_pan.pos" in wp: wp["shoulder_pan.pos"] += pan_offset
                            if "shoulder_lift.pos" in wp: wp["shoulder_lift.pos"] += lift_offset

                    # Replace playback waypoints with adjusted ones
                    playback_waypoints = current_path
                    robot.bus.enable_torque()
                    mode = "PLAYBACK"
                    playback_index = 0
                    main.sub_idx = 0 # Reset interpolation index
                    last_move_time = time.time()
                
                if key == ord('t'):
                    mode = "TEACHING"
                    robot.bus.disable_torque()

            elif mode == "PLAYBACK":
                cur_time = time.time()
                if cur_time - last_move_time > time_between_waypoints/trajectory_steps:
                    if playback_index < len(waypoints) - 1:
                        # Smooth Trajectory Logic: Linear Interpolation (LERP)
                        # We are moving from waypoints[playback_index] to [playback_index+1]
                        # but in small 'trajectory_steps'
                        start_wp = waypoints[playback_index]
                        end_wp = waypoints[playback_index+1]
                        
                        # Use a sub-index for the interpolation
                        if not hasattr(main, 'sub_idx'): main.sub_idx = 0
                        
                        ratio = main.sub_idx / trajectory_steps
                        interp_wp = {}
                        for k in start_wp:
                            if isinstance(start_wp[k], (int, float)):
                                interp_wp[k] = start_wp[k] + (end_wp[k] - start_wp[k]) * ratio
                            else:
                                interp_wp[k] = start_wp[k]

                        robot.send_action(interp_wp)
                        main.sub_idx += 1
                        
                        if main.sub_idx > trajectory_steps:
                            main.sub_idx = 0
                            playback_index += 1
                        
                        last_move_time = cur_time
                    elif playback_index == len(waypoints) - 1:
                        # Reach the final point
                        robot.send_action(waypoints[-1])
                        print("Folding completed.")
                        robot.bus.disable_torque()
                        mode = "TEACHING"
                        playback_index = 0

            cv2.imshow("Cloth folding Bot v2 (YOLO)", frame)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.bus.disable_torque()
        robot.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()
