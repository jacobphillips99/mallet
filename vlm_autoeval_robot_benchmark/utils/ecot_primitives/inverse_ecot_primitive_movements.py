import json
import os

import numpy as np

from vlm_autoeval_robot_benchmark.utils.ecot_primitives.ecot_primitive_movements import (
    DIRECTION_NAMES,
    REVERSE_DIRECTION_NAMES,
)

# Construct the path to the action_bounds.json file
current_dir = os.path.dirname(__file__)
action_bounds_path = os.path.join(current_dir, "action_bounds.json")

# Load the action bounds from the JSON file
ACTION_BOUNDS: dict[str, dict[str, dict[str, float]]] = json.load(open(action_bounds_path))


def normalize_action_to_bounds(k: str, pred_direction: str, pred_magnitude: float) -> float:
    # find the signed direction, ie "forward" -> 1, "backward" -> -1
    signed_direction = REVERSE_DIRECTION_NAMES[k][pred_direction]
    # find the magnitude w.r.t to the action bounds in the correct direction
    action_bounds = ACTION_BOUNDS[k]["actions"]
    half_range = action_bounds["interdecile_range"] / 2
    scaled_magnitude = pred_magnitude * half_range
    # either add or subtract the scaled magnitude from the median
    return action_bounds["median"] + signed_direction * scaled_magnitude


def text_to_move_vector(
    payload: dict[str, tuple[str, float, str]],
) -> np.ndarray:
    # payload is a dict of the form {'axis': ['direction', magnitude, 'explanation']}
    # except gripper, which is just ['direction', 'explanation']
    move_vector = np.zeros(7)
    for i, k in enumerate(DIRECTION_NAMES.keys()):
        if k not in payload:
            raise ValueError(
                f"Axis {k} not found in payload! Payload: {json.dumps(payload, indent=4)}"
            )

        # split payload into direction, magnitude, explanation
        pred_direction, pred_magnitude, _ = payload[k]
        if k == "gripper":
            # gripper ignores magnitude, just use the signed direction
            maybe_pred_direction = REVERSE_DIRECTION_NAMES[k].get(pred_direction)
            if maybe_pred_direction is None:
                raise ValueError(
                    f"Direction {maybe_pred_direction} not found in REVERSE_DIRECTION_NAMES for axis {k}: {REVERSE_DIRECTION_NAMES[k]}"
                )
            # clip the gripper to 0-1
            move_vector[i] = np.clip(maybe_pred_direction, 0, 1)
        else:
            # early exit on None
            if pred_direction.lower() in ["none", None]:
                move_vector[i] = 0
            else:
                if pred_direction not in REVERSE_DIRECTION_NAMES[k]:
                    raise ValueError(
                        f"Direction {pred_direction} not found in REVERSE_DIRECTION_NAMES for axis {k}: {REVERSE_DIRECTION_NAMES[k]}"
                    )
                move_vector[i] = normalize_action_to_bounds(k, pred_direction, pred_magnitude)
    return move_vector
