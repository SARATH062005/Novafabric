import time
import json
import os
import argparse
import cv2
import numpy as np
import threading
from pathlib import Path

from ultralytics import YOLO

import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""
os.environ["QT_QPA_PLATFORM"] = "xcb"


from PyQt5 import QtWidgets, QtGui, QtCore

from lerobot.robots.so_follower.so_follower import SOFollower, SOFollowerRobotConfig


# -------------------- VISION -------------------- #
def detect_white_tray(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 140])
    upper_white = np.array([180, 80, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 10000:
            return cv2.boundingRect(largest)
    return None


def detect_cloth(frame, tray_roi=None):
    detect_frame = frame
    offset_x, offset_y = 0, 0

    if tray_roi is not None:
        tx, ty, tw, th = tray_roi
        detect_frame = frame[ty:ty+th, tx:tx+tw]
        offset_x, offset_y = tx, ty

    hsv = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 180])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 4000:
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10']/M['m00']) + offset_x
                cy = int(M['m01']/M['m00']) + offset_y
                return True, (cx, cy)

    return False, (None, None)


# -------------------- PLAYBACK THREAD -------------------- #
class PlaybackThread(threading.Thread):
    def __init__(self, robot, waypoints, speed):
        super().__init__()
        self.robot = robot
        self.waypoints = waypoints
        self.speed = speed
        self._stop_event = threading.Event()

    def run(self):
        print("Playback started")
        start_time = time.time()

        for i, wp in enumerate(self.waypoints):
            if self._stop_event.is_set():
                print("Playback stopped")
                return

            action = {k: v for k, v in wp.items() if k.endswith(".pos")}

            if action:
                self.robot.send_action(action)

            elapsed = time.time() - start_time
            expected = (i + 1) * self.speed
            time.sleep(max(0, expected - elapsed))

        print("Playback finished")

    def stop(self):
        self._stop_event.set()


# -------------------- PYQT WINDOW -------------------- #
INDUSTRIAL_THEME = """
QWidget {
    background-color: #1A1A1A;
    color: #F0F0F0;
    font-family: 'Segoe UI', 'Roboto', 'Inter', 'Noto Color Emoji', 'Apple Color Emoji', 'Segoe UI Emoji', sans-serif;
}
QLabel#headerTitle {
    font-size: 26px;
    font-weight: 800;
    color: #00E676;
    background: transparent;
}
QLabel#timeLabel {
    font-size: 20px;
    font-weight: bold;
    color: #B0BEC5;
    background: transparent;
}
QLabel#statusLabel {
    font-size: 16px;
    font-weight: 600;
    color: #FFFFFF;
    background-color: #2D2D2D;
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #424242;
}
QComboBox {
    font-size: 16px;
    font-weight: 600;
    background-color: #2D2D2D;
    color: #FFFFFF;
    border: 1px solid #424242;
    border-radius: 6px;
    padding: 6px 15px;
}
QComboBox::drop-down {
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #2D2D2D;
    color: #FFFFFF;
    selection-background-color: #00E676;
    selection-color: #1A1A1A;
}
QLabel#statLabel {
    font-size: 18px;
    color: #00E676;
    background-color: #252525;
    padding: 18px;
    border-radius: 8px;
    border: 1px solid #333333;
    font-weight: 600;
}
QLabel#cameraFeed {
    border: 2px solid #333333;
    border-radius: 8px;
    background-color: #000000;
}
"""

class CameraWindow(QtWidgets.QWidget):
    def __init__(self, robot, args, robot_online=True):
        super().__init__()
        self.robot = robot
        self.args = args
        self.robot_online = robot_online

        self.cap = cv2.VideoCapture(args.camera_index)

        self.setWindowTitle("NOVAFAB Robot Dashboard")
        self.resize(1150, 700)
        self.setStyleSheet(INDUSTRIAL_THEME)

        # ---------------- STATE ---------------- #
        self.cycle_count = 0
        self.cycle_id = 1
        self.cycle_start_time = None

        # ---------------- TOP BAR ---------------- #
        self.time_label = QtWidgets.QLabel()
        self.time_label.setObjectName("timeLabel")
        
        self.robot_status = QtWidgets.QLabel("🤖 Robot: Idle")
        self.robot_status.setObjectName("statusLabel")
        
        self.cloth_selector_layout = QtWidgets.QHBoxLayout()
        self.cloth_label = QtWidgets.QLabel("👕 Cloth: ")
        self.cloth_label.setObjectName("statusLabel")
        self.cloth_selector = QtWidgets.QComboBox()
        self.cloth_selector.addItems(["None", "Kerchief", "T-Shirt", "Towel", "Pants"])
        self.cloth_selector.setFocusPolicy(QtCore.Qt.NoFocus)
        self.cloth_selector_layout.addWidget(self.cloth_label)
        self.cloth_selector_layout.addWidget(self.cloth_selector)

        self.set_home_btn = QtWidgets.QPushButton("🏠 Set Home")
        self.set_home_btn.setStyleSheet("font-size:16px; font-weight:bold; background-color:#2D2D2D; color:#00E676; padding:6px 15px; border-radius:6px; border:1px solid #424242;")
        self.set_home_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.set_home_btn.clicked.connect(self.save_home_position)
        
        dashboard_label = QtWidgets.QLabel("🤖 NOVAFAB Dashboard")
        dashboard_label.setObjectName("headerTitle")

        self.health_status = QtWidgets.QLabel("🟢 Online" if self.robot_online else "🔴 Offline")
        self.health_status.setObjectName("statusLabel")
        if not self.robot_online:
            self.health_status.setStyleSheet("color: #FF5252;")

        top_row_1 = QtWidgets.QHBoxLayout()
        top_row_1.addWidget(dashboard_label)
        top_row_1.addSpacing(15)
        top_row_1.addWidget(self.health_status)
        top_row_1.addStretch()
        top_row_1.addWidget(self.time_label)

        top_row_2 = QtWidgets.QHBoxLayout()
        top_row_2.addLayout(self.cloth_selector_layout)
        top_row_2.addSpacing(15)
        top_row_2.addWidget(self.set_home_btn)
        top_row_2.addStretch()
        top_row_2.addWidget(self.robot_status)

        top_layout = QtWidgets.QVBoxLayout()
        top_layout.addLayout(top_row_1)
        top_layout.addLayout(top_row_2)

        # ---------------- CAMERA ---------------- #
        self.image_label = QtWidgets.QLabel()
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_label.setObjectName("cameraFeed")

        # ---------------- YOLO MODEL ---------------- #
        self.yolo_model = YOLO("yolo11s-seg.pt")

        # ---------------- RIGHT PANEL ---------------- #
        self.cycle_time_label = QtWidgets.QLabel("⏳ Cycle Time: 0.0 s")
        self.cycle_time_label.setObjectName("statLabel")
        
        self.cycle_id_label = QtWidgets.QLabel("🔢 Cycle ID: 1")
        self.cycle_id_label.setObjectName("statLabel")
        
        self.cycle_done_label = QtWidgets.QLabel("✅ Cycle Completed: 0")
        self.cycle_done_label.setObjectName("statLabel")

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.cycle_time_label)
        right_layout.addSpacing(10)
        right_layout.addWidget(self.cycle_id_label)
        right_layout.addSpacing(10)
        right_layout.addWidget(self.cycle_done_label)
        right_layout.addStretch()

        # ---------------- MAIN LAYOUT ---------------- #
        middle_layout = QtWidgets.QHBoxLayout()
        middle_layout.addWidget(self.image_label)
        middle_layout.addLayout(right_layout)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(middle_layout)

        self.setLayout(main_layout)

        # ---------------- TIMERS ---------------- #
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.clock_timer = QtCore.QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

        self.cycle_timer = QtCore.QTimer()
        self.cycle_timer.timeout.connect(self.update_cycle_time)
        self.cycle_timer.start(100)

        self.play_thread = None

        # IMPORTANT: enable keyboard focus
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    # ---------------- CLOCK ---------------- #
    def update_clock(self):
        self.time_label.setText(time.strftime("%H:%M:%S"))

    # ---------------- CAMERA ---------------- #
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()

        rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(qt_img)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    # ---------------- CYCLE TIMER ---------------- #
    def update_cycle_time(self):
        if self.cycle_start_time:
            elapsed = time.time() - self.cycle_start_time
            self.cycle_time_label.setText(f"⏳ Cycle Time: {elapsed:.1f} s")

    # ---------------- KEYBOARD CONTROL ---------------- #
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.complete_cycle()

        elif event.key() == QtCore.Qt.Key_0:
            self.start_sequence("home_position.json")

        elif event.key() == QtCore.Qt.Key_1:
            self.start_sequence("folding_waypoints_1_seq.json")

        elif event.key() == QtCore.Qt.Key_2:
            self.start_sequence("folding_waypoints_2_seq.json")

    # ---------------- SET HOME POSITION ---------------- #
    def save_home_position(self):
        if not self.robot_online:
            self.robot_status.setText("🤖 Error: Offline")
            return
            
        try:
            obs = self.robot.get_observation()
            # Extract just the position metrics to save
            home_pos_dict = {k: float(v) for k, v in obs.items() if k.endswith('.pos')}
            
            with open("home_position.json", "w") as f:
                json.dump(home_pos_dict, f, indent=4)
                
            self.robot_status.setText("🤖 Home Position Saved!")
        except Exception as e:
            self.robot_status.setText("🤖 Error: Could not save")
            print(f"Error saving home position: {e}")

    # ---------------- COMPLETE CYCLE ---------------- #
    def complete_cycle(self):
        self.cycle_count += 1
        self.cycle_id += 1

        self.cycle_done_label.setText(f"✅ Cycle Completed: {self.cycle_count}")
        self.cycle_id_label.setText(f"🔢 Cycle ID: {self.cycle_id}")

        self.cycle_start_time = time.time()

    # ---------------- PLAYBACK ---------------- #
    def start_sequence(self, file):
        if not self.robot_online:
            self.robot_status.setText("🤖 Error: Offline")
            return
            
        self.robot_status.setText("🤖 Robot: Playing")

        self.cycle_start_time = time.time()

        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.stop()
            self.play_thread.join()

        if not os.path.exists(file):
            self.robot_status.setText("🤖 Robot: File Missing")
            return

        with open(file, "r") as f:
            waypoints = json.load(f)
            
        if isinstance(waypoints, dict):
            waypoints = [waypoints]

        self.play_thread = PlaybackThread(self.robot, waypoints, self.args.speed)
        self.play_thread.start()

    # ---------------- CLEANUP ---------------- #
    def closeEvent(self, event):
        if self.play_thread:
            self.play_thread.stop()
            self.play_thread.join()

        self.cap.release()
        self.robot.bus.disable_torque()
        self.robot.disconnect()

        event.accept()


# -------------------- MAIN -------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--accel", type=int, default=220)
    parser.add_argument("--stiffness", type=int, default=48)
    parser.add_argument("--gripper-torque", type=int, default=800)
    parser.add_argument("--motor-torque", type=int, default=500)
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--robot-id", type=str, default="my_leader_arm3")
    parser.add_argument("--use-vision", action="store_true")
    parser.add_argument("--camera-index", type=int, default=2)

    args = parser.parse_args()

    print("Initializing robot...")

    config = SOFollowerRobotConfig(
        port=args.port,
        id=args.robot_id,
        use_degrees=True,
        disable_torque_on_disconnect=True,
        calibration_dir=Path("."),
    )

    robot = SOFollower(config)

    robot_online = True
    try:
        robot.connect(calibrate=True)

        for motor in robot.bus.motors:
            try:
                robot.bus.write("Acceleration", motor, args.accel)
                robot.bus.write("P_Coefficient", motor, args.stiffness)

                torque = args.gripper_torque if motor == "gripper" else args.motor_torque

                robot.bus.disable_torque(motor)
                robot.bus.write("Max_Torque_Limit", motor, torque)
                robot.bus.enable_torque(motor)

            except Exception as e:
                print(f"Motor setup warning: {e}")

        robot.bus.enable_torque()

    except Exception as e:
        print(f"Robot init failed: {e}")
        robot_online = False

    # Start PyQt app
    app = QtWidgets.QApplication([])

    window = CameraWindow(robot, args, robot_online)
    window.show()

    app.exec_()


if __name__ == "__main__":
    main()