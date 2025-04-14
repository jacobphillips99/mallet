GRIPPER_INDEX = 6
GRIPPER_OPEN_THRESHOLD = 0.95


def get_gripper_position(gripper_state: float) -> str:
    return "OPEN" if gripper_state > GRIPPER_OPEN_THRESHOLD else "CLOSED"
