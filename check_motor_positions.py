import time
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

def main():
    # Use raw bus to avoid calibration checks
    bus = FeetechMotorsBus(
        port="/dev/ttyACM0",
        motors={
            "motor1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "motor2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "motor3": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "motor4": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "motor5": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "motor6": Motor(6, "sts3215", MotorNormMode.RANGE_M100_100),
        }
    )
    try:
        bus.connect()
        print("Connected. Checking motor positions (0-4095 range):")
        for motor_name in bus.motors:
            try:
                # Read register 56 (Present_Position) directly or via the bus.read
                val = bus.read("Present_Position", motor_name)
                print(f"{motor_name} (id {bus.motors[motor_name].id}): {val}")
            except Exception as e:
                print(f"Error reading {motor_name}: {e}")
    finally:
        bus.disconnect()

if __name__ == "__main__":
    main()
