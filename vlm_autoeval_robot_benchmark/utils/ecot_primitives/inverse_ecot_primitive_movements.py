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
    # converts text direction and magnitude to normalized action value
    # find the signed direction, ie "forward" -> 1, "backward" -> -1
    signed_direction = REVERSE_DIRECTION_NAMES[k][pred_direction]
    # find the magnitude w.r.t to the action bounds in the correct direction
    action_bounds = ACTION_BOUNDS[k]["actions"]
    half_range = action_bounds["interdecile_range"] / 2
    scaled_magnitude = pred_magnitude * half_range
    # either add or subtract the scaled magnitude from the median
    return action_bounds["median"] + signed_direction * scaled_magnitude


def unnormalize_bounded_action(k: str, normalized_action_value: float) -> tuple[float, str]:
    # converts normalized action value to text direction and magnitude
    if k == "gripper":
        # gripper is a special case, it's a binary action
        magnitude = 1.0 if normalized_action_value > 0.95 else 0.0
        direction_str = "open gripper" if magnitude == 1.0 else "close gripper"
        return magnitude, direction_str

    # find the difference from the median, unnormalize according to range
    median_diff = normalized_action_value - ACTION_BOUNDS[k]["actions"]["median"]
    half_range = ACTION_BOUNDS[k]["actions"]["interdecile_range"] / 2
    direction_value = int(np.sign(median_diff))

    # handle middle-ground None case
    if direction_value == 0:
        return 0.0, "None"

    # handle null roll case
    if direction_value not in DIRECTION_NAMES[k]:
        direction_str = "None"
    else:
        direction_str = str(DIRECTION_NAMES[k][direction_value])
    magnitude = abs(median_diff) / half_range
    return magnitude, direction_str


def text_to_move_vector(
    payload: dict[str, tuple[str | None, float, str]],
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
            if isinstance(pred_direction, str) and pred_direction.lower() == "none":
                # we don't want None outputs for gripper but handle it just in case
                pred_direction = None
            maybe_pred_direction = REVERSE_DIRECTION_NAMES[k].get(pred_direction)
            if maybe_pred_direction is None:
                raise ValueError(
                    f"Direction {maybe_pred_direction} not found in REVERSE_DIRECTION_NAMES for axis {k}: {REVERSE_DIRECTION_NAMES[k]}"
                )
            # clip the gripper to 0-1
            move_vector[i] = np.clip(maybe_pred_direction, 0, 1)
        else:
            # early exit on None
            if (
                isinstance(pred_direction, str) and pred_direction.lower() == "none"
            ) or pred_direction is None:
                move_vector[i] = 0
            else:
                if pred_direction not in REVERSE_DIRECTION_NAMES[k]:
                    raise ValueError(
                        f"Direction {pred_direction} not found in REVERSE_DIRECTION_NAMES for axis {k}: {REVERSE_DIRECTION_NAMES[k]}"
                    )
                move_vector[i] = normalize_action_to_bounds(k, pred_direction, pred_magnitude)
    return move_vector
