"""
Annotated and reworked copy of the Embodied Chain of Thought (ECoT) paper's "primitive movements",
which turns 7-dof robot actions into a text-based descriptions.
https://github.com/MichalZawalski/embodied-CoT/blob/main/scripts/generate_embodied_data/primitive_movements.py
"""

import copy
import typing as t

import numpy as np

# Direction mappings for each of the 7 DOF
# Indices represent:
# 0-2: x,y,z translations (forward/backward, left/right, up/down)
# 3: tilt (pitch)
# 4: roll (empty in this implementation, as 3 and 4 are treated as equivalent)
# 5: rotation (yaw)
# 6: gripper state
DIRECTION_NAMES: dict[str, dict[int, t.Optional[str]]] = {
    "x": {-1: "backward", 0: None, 1: "forward"},  # x-axis
    "y": {-1: "right", 0: None, 1: "left"},  # y-axis
    "z": {-1: "down", 0: None, 1: "up"},  # z-axis
    "tilt": {-1: "tilt down", 0: None, 1: "tilt up"},  # pitch
    "roll": {},  # roll (unused)
    "rotation": {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},  # yaw
    "gripper": {-1: "close gripper", 0: None, 1: "open gripper"},  # gripper
}
REVERSE_DIRECTION_NAMES = {k: {v: k for k, v in v.items()} for k, v in DIRECTION_NAMES.items()}


def describe_move(move_vec: np.ndarray) -> str:
    """
    Converts a 7-DOF movement vector to a human-readable text description.

    Args:
        move_vec: A vector of shape (7,) with values {-1, 0, 1} indicating movement direction
                 for each degree of freedom

    Returns:
        A string description of the movement
    """
    names = copy.deepcopy(DIRECTION_NAMES)

    # Get text descriptions for x, y, z movements
    xyz_move_with_none: list[str | None] = [
        names[ax][move_vec[i]] for i, ax in enumerate(["x", "y", "z"])
    ]
    xyz_move: list[str] = [m for m in xyz_move_with_none if m is not None]

    # Start building the description with translations
    if len(xyz_move) != 0:
        description = "move " + " ".join(xyz_move)
    else:
        description = ""

    # Handle roll and pitch together
    if move_vec[3] == 0:
        move_vec[3] = move_vec[4]  # identify rolling and pitching as equivalent

    # Add tilt (pitch) description
    if move_vec[3] != 0:
        if len(description) > 0:
            description = description + ", "
        this_move = names["tilt"][move_vec[3]]
        if this_move is not None:
            description = description + this_move

    # Add rotation (yaw) description
    if move_vec[5] != 0:
        if len(description) > 0:
            description = description + ", "
        this_move = names["rotation"][move_vec[5]]
        if this_move is not None:
            description = description + this_move

    # Add gripper state description
    if move_vec[6] != 0:
        if len(description) > 0:
            description = description + ", "
        this_move = names["gripper"][move_vec[6]]
        if this_move is not None:
            description = description + this_move

    # Default when no movement
    if len(description) == 0:
        description = "stop"

    return description


def classify_movement(move: np.ndarray, threshold: float = 0.03) -> tuple[str, np.ndarray]:
    """
    Converts a sequence of robot states into a movement primitive description.

    Args:
        move: A sequence of robot states, typically a trajectory segment
        threshold: Movement threshold below which movements are considered negligible

    Returns:
        A tuple containing:
            - Text description of the movement
            - Quantized movement vector with elements in {-1, 0, 1}

    Edited to permit predicting on just an action instead of a difference between two states
    """
    if len(move.shape) == 1:
        # just use the action as the diff in states
        diff = move
    else:
        # Calculate difference between end and start states
        diff = move[-1] - move[0]

    # Normalize translation movements if they're too large
    if np.sum(np.abs(diff[:3])) > 3 * threshold:
        diff[:3] *= 3 * threshold / np.sum(np.abs(diff[:3]))

    # Scale rotation components differently
    diff[3:6] /= 10

    # Quantize movements to {-1, 0, 1} based on threshold
    move_vec = 1 * (diff > threshold) - 1 * (diff < -threshold)

    return describe_move(move_vec), move_vec


# Dictionary to store observed actions for each movement type
move_actions: dict[str, list[np.ndarray]] = dict()


def get_move_primitives_episode(
    episode: dict, threshold: float = 0.03, window: int = 4
) -> list[tuple[str, np.ndarray]]:
    """
    Extracts movement primitives from a full episode.

    Args:
        episode: Dictionary containing episode data with steps, observations, and actions

    Returns:
        List of (movement_description, movement_vector) tuples for each step
    """
    steps = list(episode["steps"])

    # Extract state and action sequences
    states = np.array([step["observation"]["state"] for step in steps])
    actions = [step["action"][:3] for step in steps]

    # Create trajectory segments and classify them
    move_trajs = [states[i : i + 4] for i in range(len(states) - 1)]
    primitives = [classify_movement(move, threshold) for move in move_trajs]
    primitives.append(primitives[-1])  # Repeat last primitive for final step

    # Store actions for each movement type
    for (move, _), action in zip(primitives, actions):
        if move in move_actions:
            move_actions[move].append(action)
        else:
            move_actions[move] = [action]

    return primitives


def get_move_primitives(episode_id: int, builder: t.Any) -> list[tuple[str, np.ndarray]]:
    """
    Retrieves movement primitives for a specific episode using a dataset builder.

    Args:
        episode_id: ID of the episode to process
        builder: Dataset builder object with as_dataset method

    Returns:
        List of movement primitives for the episode
    """
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))

    return get_move_primitives_episode(episode)
