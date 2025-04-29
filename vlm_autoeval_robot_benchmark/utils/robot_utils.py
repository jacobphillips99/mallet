import numpy as np

GRIPPER_INDEX = -1
GRIPPER_OPEN_THRESHOLD = 0.1


def get_gripper_position(gripper_state: float) -> str:
    return "OPEN" if gripper_state > GRIPPER_OPEN_THRESHOLD else "CLOSED"


"""
        import numpy as np
        import torch  # lazy
        from src.utils.geometry import mat2euler, quat2mat

        # preprocess proprio from 8D to 7D
        proprio = obs_dict["proprio"]
        assert len(proprio) == 8, "original proprio should be size 8"
        rm_bridge = quat2mat(proprio[3:7])
        # EE pose in Bridge data was relative to a top-down pose, instead of robot base
        default_rot = np.array(
            [[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]
        )  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        rpy_bridge_converted = mat2euler(rm_bridge @ default_rot.T)
        gripper = proprio[7]
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                [gripper],
            ]
        )

        # normalize proprios - gripper opening is normalized
        eps = 1e-8
        if self.config["action_normalization_type"] == "bounds":
            proprio = (
                2
                * (raw_proprio - self.dataset_stats["proprio"]["q01"])
                / (
                    self.dataset_stats["proprio"]["q99"]
                    - self.dataset_stats["proprio"]["q01"]
                    + eps
                )
                - 1
            )
            proprio = np.clip(proprio, -1, 1)
        elif self.config["action_normalization_type"] == "normal":
            proprio = (raw_proprio - self.dataset_stats["proprio"]["mean"]) / (
                self.dataset_stats["proprio"]["std"] + eps
            )"""

gripper_dataset_stats = {
    "proprio": {
        "q01": 0.0,
        "q99": 1.0,
        "mean": 0.5764579176902771,
        "std": 0.49737006425857544,
    }
}


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
