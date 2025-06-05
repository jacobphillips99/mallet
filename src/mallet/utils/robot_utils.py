import numpy as np

GRIPPER_INDEX = -1
GRIPPER_OPEN_THRESHOLD = 0.1


def get_gripper_position(gripper_state: float) -> str:
    return "OPEN" if gripper_state > GRIPPER_OPEN_THRESHOLD else "CLOSED"


def norm_step(raw_proprio: np.ndarray, norm_type: str, dataset_stats: dict) -> np.ndarray:
    # normalize proprios - gripper opening is normalized
    eps = 1e-8
    if norm_type == "bounds":
        q01 = dataset_stats["proprio"]["q01"]
        q99 = dataset_stats["proprio"]["q99"]
        proprio = 2 * (raw_proprio - q01) / (q99 - q01 + eps) - 1
        proprio = np.clip(proprio, -1, 1)
    elif norm_type == "normal":
        mean = dataset_stats["proprio"]["mean"]
        std = dataset_stats["proprio"]["std"]
        proprio = (raw_proprio - mean) / (std + eps)
    return proprio


def unnorm_step(proprio: np.ndarray, norm_type: str, dataset_stats: dict) -> np.ndarray:
    if norm_type == "bounds":
        q01 = dataset_stats["proprio"]["q01"]
        q99 = dataset_stats["proprio"]["q99"]
        raw_proprio = 2 * (proprio + 1) * (q99 - q01) / 2 + q01
    elif norm_type == "normal":
        mean = dataset_stats["proprio"]["mean"]
        std = dataset_stats["proprio"]["std"]
        raw_proprio = proprio * std + mean
    return raw_proprio
