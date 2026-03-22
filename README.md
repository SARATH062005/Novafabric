# Novafabric: Robotic Cloth Folding

**Novafabric** is an AI-powered robotic cloth-folding system built on the **LeRobot** framework. It uses an **SO-100/10X** robotic arm and **YOLOv11** segmentation to achieve high-precision autonomous folding.

## 🚀 Features

- **YOLOv11 Segmentation**: Real-time detection and mask segmentation of the fabric for precise grasping.
- **Trajectory Teaching**: Continuous path recording during human teaching—no more waiting for the arm to "stop" to save a point.
- **Dynamic Vision Offset**: Automatically adjusts the entire folding trajectory based on the cloth's detected position in the frame.
- **High-Speed Playback**: Optimized motor acceleration and path following for fluid, human-like motion.
- **Hardware Agnostic**: Tested with Feetech STS3215 motors on the SO-follower architecture.

## 🛠️ Requirements

- [LeRobot](https://github.com/huggingface/lerobot)
- `ultralytics` (YOLOv11-seg)
- `opencv-python`
- `numpy`
- Conda Environment: `lerobot`

## 📦 Project Structure

- `cloth_folding_cv.py`: Main application script (Teaching & Playback with Vision).
- `camera.py`: Camera diagnostic and property optimization tool.
- `folding_waypoints.json`: Saved path data.
- `my_leader_arm3.json`: Optimized calibration configuration.
- `check_motor_positions.py`: Troubleshooting tool for motor range and centering.

## 🕹️ Instructions

1. **Activate Environment**:
   ```bash
   conda activate lerobot
   ```
2. **Run Calibration (if new arm)**:
   The script will prompt to calibrate or use existing `my_leader_arm3.json`.
3. **Teaching Mode**:
   - Move the arm by hand to record the path.
   - `[G]`: Toggle gripper.
   - `[Backspace]`: Undo last point.
   - `[ENTER]`: Manual save point.
4. **Playback Mode**:
   - `[P]`: Start autonomous folding.
   - The robot will wait until the cloth is detected by the camera before starting the sequence.

---
Developed as part of the Novafabric Autonomous Robotics project.
