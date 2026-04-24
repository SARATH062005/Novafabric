import pybullet as p
import pybullet_data
import time
import numpy as np

URDF_PATH = "so_100_arm_5dof.urdf"   # make sure this URDF file is in your folder
ROBOT_START_POS = [0, 0, 0]
ROBOT_START_ORN = p.getQuaternionFromEuler([0, 0, 0])
GRIPPER_LINK_NAME = "Moving_Jaw"

# ----------------------------
# Setup environment
# ----------------------------
p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")

# ----------------------------
# Load robot
# ----------------------------
robot_id = p.loadURDF(URDF_PATH, ROBOT_START_POS, ROBOT_START_ORN, useFixedBase=True)

# ----------------------------
# Create sliders for each joint
# ----------------------------
joint_sliders = {}
for j in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, j)
    joint_name = joint_info[1].decode("utf-8")
    joint_type = joint_info[2]

    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        lower = joint_info[8]
        upper = joint_info[9]
        if lower < -3.14: lower = -3.14
        if upper > 3.14: upper = 3.14

        slider = p.addUserDebugParameter(joint_name, lower, upper, 0)
        joint_sliders[j] = slider

print("Control your robot with the sliders in the left panel.")

# ----------------------------
# Helper: get gripper link index
# ----------------------------
gripper_link_index = None
for j in range(p.getNumJoints(robot_id)):
    if p.getJointInfo(robot_id, j)[12].decode("utf-8") == GRIPPER_LINK_NAME:
        gripper_link_index = j
        break

if gripper_link_index is None:
    raise ValueError("Gripper link not found in URDF")

# ----------------------------
# Simulation loop with camera on gripper
# ----------------------------
while True:
    # Read sliders and apply joint control
    for j, slider in joint_sliders.items():
        target = p.readUserDebugParameter(slider)
        p.setJointMotorControl2(robot_id,
                                j,
                                p.POSITION_CONTROL,
                                targetPosition=target,
                                force=200)

    # Get gripper state
    link_state = p.getLinkState(robot_id, gripper_link_index)
    gripper_pos = link_state[4]  # world position
    gripper_orn = link_state[5]  # world orientation quaternion

    # Compute camera orientation
    rot_matrix = np.array(p.getMatrixFromQuaternion(gripper_orn)).reshape(3,3)
    forward_vec = rot_matrix[:,0]  # x-axis forward in link frame
    up_vec = rot_matrix[:,2]       # z-axis up in link frame
    cam_target = gripper_pos + forward_vec * 0.1  # look slightly ahead
    cam_pos = gripper_pos          # camera at gripper

    # Update camera in GUI
    view_matrix = p.computeViewMatrix(cam_pos, cam_target, up_vec)
    p.getCameraImage(320, 240, viewMatrix=view_matrix)

    # Step simulation
    p.stepSimulation()
    time.sleep(1./240.)
