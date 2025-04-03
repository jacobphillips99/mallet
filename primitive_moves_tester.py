"""
Test script to validate ECoT primitive movements and their inverse transformations.
This script covers two main testing scenarios:
1. Comparing VLM image interpretations with ECOT action vector descriptions
2. Testing the round-trip conversion: action → text → action

The action vector is a 7D vector that describes the robot's movement.
# Direction mappings for each of the 7 DOF
# Indices represent:
# 0-2: x,y,z translations (forward/backward, left/right, up/down)
# 3: tilt (pitch)
# 4: roll (empty in this implementation, as 3 and 4 are treated as equivalent)
# 5: rotation (yaw)
# 6: gripper state
"""

"""
There's lots of context in the AutoEval repo that is relevant to this script.
We're going to focus on the `drawer` environment out of [drawer, sink]
Here's some notes on the environment:

https://github.com/zhouzypaul/auto_eval/blob/c64202a120adc405017baeb03ae8b76f3d3ddb1c/scripts/configs/eval_config.py#L191
workspace_bounds=dict(
        x=[0.12, float("inf")],  # edge of table
        y=[-float("inf"), float("inf")],
        z=[0, float("inf")],
    ),
    # x is towards front (slight right) wall, y is towards left wall, z is up
    failure_conditions=[
        {
            "x": lambda x: x >= 0.43,
            "y": lambda y: True,
            "z": lambda z: z <= 0.03,
        },  # robot pushing the micromove and falling
        {
            "x": lambda x: True,
            "y": lambda y: True,
            "z": lambda z: z <= 0,
        },  # robot falling on the table somewhere
        # {
        #     "x": lambda x: x >= 0.382,
        #     "y": lambda y: y >= 0.01,
        #     "z": lambda z: z <= 0.07,
        # },  # robot arm stuck behind drawer handle so it's hard to get back to reset
    ],
    stuck_conditions=[
        # {
        #     "x": lambda x: 0.27 <= x <= 0.3,
        #     "y": lambda y: -0.05 <= y <= 0.05,
        #     "z": lambda z: 0.02 <= z <= 0.063,
        # },  # handle in drawer handle
    ],

- workspace_bounds: goes into a ClipActionBoxBoundary object from manipulator_gym (just keeps action in global bounds)
- failure_conditions: runs at end of episode. raised RobotFailure if e.g. robot falls on the table somewhere. Runs against the 3D xyz position as basically `all([condition[axis](xyz[axis]) for axis in ['x', 'y', 'z']])`
- stuck_conditions: same as failure, but for "stuck" positions like behind the drawer handle
"""

import numpy as np
import asyncio
from pathlib import Path
import base64
import typing as t
import json

from vlm_autoeval_robot_benchmark.utils.ecot_primitives.ecot_primitive_movements import describe_move, classify_movement
from vlm_autoeval_robot_benchmark.utils.ecot_primitives.inverse_ecot_primitive_movements import get_action_from_description
from vlm_autoeval_robot_benchmark.models.vlm import VLM, VLMRequest, ImageInput

MODEL = "gpt-4o-mini"  # Can be changed to any supported model

# Mock data for testing
# In reality, this would be loaded from a dataset
MOCK_TEST_SAMPLES = [
    {
        "image_path": "assets/robot_move_forward.png",  # This is a mock path
        "action_vector": np.array([1, 0, 0, 0, 0, 0, 0]),  # Move forward
        "expected_description": "move forward"
    },
    {
        "image_path": "assets/robot_move_left.png",  # This is a mock path
        "action_vector": np.array([0, 1, 0, 0, 0, 0, 0]),  # Move left
        "expected_description": "move left" 
    },
    {
        "image_path": "assets/robot_open_gripper.png",  # This is a mock path
        "action_vector": np.array([0, 0, 0, 0, 0, 0, 1]),  # Open gripper
        "expected_description": "open gripper"
    }
]

async def test_vlm_vs_ecot():
    """
    Test Scenario 1: Compare VLM image interpretations with ECOT action vector descriptions.
    For each test sample:
    1. Feed the image to a VLM to get a text description
    2. Convert the action vector to a text description using ECOT
    3. Compare the two descriptions
    """
    print("=== Test Scenario 1: VLM(image) vs ECOT(action vector) ===\n")
    
    # Initialize VLM
    vlm = VLM()
    
    for i, sample in enumerate(MOCK_TEST_SAMPLES):
        print(f"Test sample {i+1}:")
        
        # Check if image exists (in a real scenario)
        image_path = Path(sample["image_path"])
        if not image_path.exists():
            print(f"  Image not found at {image_path}. Using mock data instead.")
            # In a real scenario, you would skip this sample or use a default image
            
            # Generate ECOT description from action vector
            ecot_description = describe_move(sample["action_vector"])
            print(f"  Action vector: {sample['action_vector']}")
            print(f"  ECOT description: '{ecot_description}'")
            print(f"  Expected VLM description: '{sample['expected_description']}'")
            print(f"  (VLM test skipped - using expected description)")
            print()
            continue
        
        # Real scenario with existing image
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Create VLM request
        vision_request = VLMRequest(
            prompt="Describe the robot movement shown in this image using simple directional terms like 'move forward', 'move left', 'tilt up', 'open gripper', etc.",
            model=MODEL,
            images=[
                ImageInput(
                    data=base64.b64encode(image_data).decode('utf-8'),
                    mime_type="image/jpeg"
                )
            ]
        )
        
        try:
            # Get description from VLM
            response = await vlm.generate(vision_request)
            vlm_description = response.text.strip()
            
            # Generate ECOT description from action vector
            ecot_description = describe_move(sample["action_vector"])
            
            print(f"  Action vector: {sample['action_vector']}")
            print(f"  ECOT description: '{ecot_description}'")
            print(f"  VLM description: '{vlm_description}'")
            
            # Compare descriptions (simple string matching for now)
            # In a real scenario, you might want to use semantic similarity
            descriptions_match = vlm_description.lower() == ecot_description.lower()
            print(f"  Descriptions match: {'✓' if descriptions_match else '✗'}")
            
        except Exception as e:
            print(f"  Error in VLM test: {e}")
        
        print()

def test_action_text_action_roundtrip():
    """
    Test Scenario 2: action → text → action round-trip conversion.
    This tests:
    1. Converting action vectors to text descriptions using ECOT
    2. Converting text descriptions back to action vectors using inverse ECOT
    3. Comparing the original and reconstructed action vectors
    """
    print("=== Test Scenario 2: action → text → action round-trip ===\n")
    
    # Test cases: Various movement vectors to test
    test_cases = [
        np.array([1, 0, 0, 0, 0, 0, 0]),   # Move forward
        np.array([0, 1, 0, 0, 0, 0, 0]),   # Move left
        np.array([0, 0, 1, 0, 0, 0, 0]),   # Move up
        np.array([0, 0, 0, 1, 0, 0, 0]),   # Tilt up
        np.array([0, 0, 0, 0, 0, 1, 0]),   # Rotate counterclockwise
        np.array([0, 0, 0, 0, 0, 0, 1]),   # Open gripper
        np.array([1, 1, 0, 0, 0, 0, 0]),   # Move forward left
        np.array([1, 0, 0, 0, 0, 0, 1]),   # Move forward, open gripper
        np.array([0, 0, 0, 0, 0, 0, 0]),   # Stop (no movement)
        np.array([-1, 0, 0, 0, 0, 0, 0]),  # Move backward
        np.array([1, -1, 1, 0, 0, -1, -1]) # Complex movement
    ]
    
    for i, original_vec in enumerate(test_cases):
        print(f"Test case {i+1}:")
        print(f"  Original vector: {original_vec}")
        
        # Convert vector to text description using ECOT
        description = describe_move(original_vec)
        print(f"  Text description: '{description}'")
        
        # Convert text description back to vector using inverse ECOT
        # Use custom_magnitude=1.0 to get normalized values
        reconstructed_action = get_action_from_description(description, custom_magnitude=1.0)
        reconstructed_vec = np.sign(reconstructed_action)  # Convert to {-1, 0, 1}
        print(f"  Reconstructed vector: {reconstructed_vec}")
        
        # Check if conversion is consistent
        is_consistent = np.array_equal(original_vec, reconstructed_vec)
        print(f"  Consistent conversion: {'✓' if is_consistent else '✗'}")
        
        if not is_consistent:
            # Identify which components differ
            diff_indices = np.where(original_vec != reconstructed_vec)[0]
            diff_components = [f"component {idx}" for idx in diff_indices]
            print(f"  Differences in: {', '.join(diff_components)}")
        
        print()

def save_mock_dataset():
    """
    Utility function to save the mock dataset to a JSON file.
    This is helpful if you want to replace it with real data later.
    """
    dataset = []
    for sample in MOCK_TEST_SAMPLES:
        # Convert numpy array to list for JSON serialization
        sample_copy = sample.copy()
        sample_copy["action_vector"] = sample["action_vector"].tolist()
        dataset.append(sample_copy)
    
    with open("mock_robot_movement_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print("Mock dataset saved to mock_robot_movement_dataset.json")

async def main():
    """Main entry point for running all tests."""
    # Test 1: VLM(image) vs ECOT(action)
    await test_vlm_vs_ecot()
    
    # Test 2: action → text → action round-trip 
    test_action_text_action_roundtrip()
    
    # Optionally save mock dataset
    # save_mock_dataset()

if __name__ == "__main__":
    asyncio.run(main())
