# Action Bounds for ECOT Primitives

This document describes the `action_bounds.json` file, which contains statistical analysis of action distributions for robotic manipulation primitives in the ECOT (Embodied Chain-of-Thought) framework.


The file contains statistical data for 7 degrees of freedom:

### Spatial Movement (3 DOF)
- **x**: Forward (+) / Backward (-) movement
- **y**: Left (+) / Right (-) movement
- **z**: Up (+) / Down (-) movement

### Rotational Movement (3 DOF)
- **tilt**: Tilt Up (+) / Down (-) rotation
- **roll**: Roll Up (+) / Down (-) rotation
- **rotation**: Rotate Counter-Clockwise (+) / Clockwise (-) rotation

### End-Effector (1 DOF)
- **gripper**: Open (+) / Close (-) gripper action

Note that gripper actions are binarized.

## Statistical Measures

For each degree of freedom, we calculate the following statistics:

| Measure | Description |
|---------|-------------|
| `mean` | Average action value across the dataset |
| `std` | Standard deviation of actions |
| `min` | Minimum observed action value |
| `max` | Maximum observed action value |
| `median` | 50th percentile (middle value) |
| `10th` | 10th percentile |
| `90th` | 90th percentile |
| `interdecile_range` | Difference between 90th and 10th percentiles |

## Collection

These numbers are pulled from the `mse-check` library and represent the distribution of actions across the provided dataset.

## Key Insights

### Movement Characteristics
- **Spatial movements** (x, y, z) have relatively small ranges (±0.03 to ±0.04)
- **Rotational movements** have larger ranges, especially `rotation` (±0.27 to ±0.29)
- Most actions are centered around zero, indicating balanced movement patterns
