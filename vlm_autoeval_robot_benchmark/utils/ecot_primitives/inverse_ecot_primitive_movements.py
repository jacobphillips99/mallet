"""
Inverse of the Embodied Chain of Thought (ECoT) primitive movements.
This module converts text-based movement descriptions back into 7-DOF robot actions.
"""

import numpy as np
import typing as t

# Import the direction names from the original module
from vlm_autoeval_robot_benchmark.utils.ecot_primitives.ecot_primitive_movements import DIRECTION_NAMES

# Create inverse mappings from text descriptions to DOF indices and values
INVERSE_DIRECTION_MAP: dict[str, tuple[int, int]] = {}
for dof_idx, direction_dict in enumerate(DIRECTION_NAMES):
    for value, name in direction_dict.items():
        if name is not None:
            INVERSE_DIRECTION_MAP[name] = (dof_idx, value)

# Default magnitudes for each DOF
DEFAULT_MAGNITUDES: list[float] = [
    0.1,  # x-axis translation
    0.1,  # y-axis translation
    0.1,  # z-axis translation
    0.3,  # pitch
    0.3,  # roll
    0.3,  # yaw
    1.0,  # gripper
]


def parse_movement_description(description: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses a text movement description into a 7-DOF movement vector and magnitude vector.
    
    Args:
        description: Human-readable movement description (e.g., "move forward left, open gripper")
    
    Returns:
        Tuple of:
            - Movement vector with elements in {-1, 0, 1}
            - Magnitude vector with default magnitudes for each DOF
    """
    # Initialize vectors with zeros
    move_vec = np.zeros(7, dtype=np.int8)
    magnitude_vec = np.array(DEFAULT_MAGNITUDES, dtype=np.float32)
    
    # Handle the special case of "stop"
    if description.strip() == "stop":
        return move_vec, magnitude_vec
    
    # Split the description into individual movement components
    components = [comp.strip() for comp in description.split(",")]
    
    # Process translation components (they start with "move")
    for i, component in enumerate(components):
        if component.startswith("move "):
            # Extract direction words after "move"
            direction_words = component[5:].split()
            
            for direction in direction_words:
                if direction in INVERSE_DIRECTION_MAP:
                    dof_idx, value = INVERSE_DIRECTION_MAP[direction]
                    move_vec[dof_idx] = value
        else:
            # Process non-translation components (rotations, tilts, gripper)
            if component in INVERSE_DIRECTION_MAP:
                dof_idx, value = INVERSE_DIRECTION_MAP[component]
                move_vec[dof_idx] = value
    
    return move_vec, magnitude_vec


def text_to_action(description: str, custom_magnitude: t.Optional[float] = None) -> np.ndarray:
    """
    Converts a text movement description to a 7-DOF robot action vector.
    
    Args:
        description: Human-readable movement description
        custom_magnitude: Optional scalar to override default magnitudes
    
    Returns:
        7-DOF action vector with appropriate magnitudes
    """
    move_vec, magnitude_vec = parse_movement_description(description)
    
    # Override magnitudes if custom_magnitude is provided
    if custom_magnitude is not None:
        magnitude_vec = np.ones_like(magnitude_vec) * custom_magnitude
    
    # Multiply direction by magnitude
    action = move_vec.astype(np.float32) * magnitude_vec
    
    return action


def get_action_from_description(description: str, custom_magnitude: t.Optional[float] = None) -> np.ndarray:
    """
    Main function to convert a text description to a robot action.
    
    Args:
        description: Movement description text
        custom_magnitude: Optional scalar to adjust the magnitude of all actions
        
    Returns:
        7-DOF robot action vector
    """
    return text_to_action(description, custom_magnitude)
