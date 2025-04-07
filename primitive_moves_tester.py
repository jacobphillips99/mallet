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
# 6: gripper state (-1 close, 1 open)

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

- workspace_bounds: goes into a ClipActionBoxBoundary object from manipulator_gym
- failure_conditions: runs at end of episode. raised RobotFailure if e.g. robot falls on the table
  somewhere. Runs against the 3D xyz position as basically:
  `all([condition[axis](xyz[axis]) for axis in ['x', 'y', 'z']])`
- stuck_conditions: same as failure, but for "stuck" positions like behind the drawer handle

We're going to focus on two `drawer` tasks: "open the drawer" and "close the drawer"

Looking at goal images, we might get an idea of if this is possible from the reset policy for the task
we care about.
E.g. We're focusing on "open the drawer", so the goal image of "close the drawer" shows the starting
position of "open the drawer". At this point, we know
"""

import asyncio
from typing import Any

from vlm_autoeval_robot_benchmark.models.vlm import VLM, create_vlm_request, parse_vlm_response

# Constants
# MODEL = "gpt-4o"
MODEL = "claude-3-5-sonnet-20241022"
# Environment descriptions
drawer_environment = """
You are looking at a wooden desk or table with a black robot arm on it.
To the left of the robot arm is a drawer with a handle.
The camera is slightly to the right of the robot.
""".strip()

sink_environment = """
You are looking at a blue sink with a yellow basket on the left.
There is an eggplant in the scene.
The camera is slightly to the right of the robot.
""".strip()

AUTOEVAL_ENVIRONMENTS = {"drawer": drawer_environment, "sink": sink_environment}

# Task descriptions
AUTOEVAL_TASKS = {
    "open_drawer": "The goal of the robot is to grab the handle of the drawer and open it.",
    "close_drawer": "The goal of the robot is to grab the handle of the drawer and close it.",
    "eggplant_to_basket": "The goal of the robot is to grab the eggplant and move it to the yellow basket.",
    "eggplant_to_sink": "The goal of the robot is to grab the eggplant and move it to the blue sink.",
}

# Image paths -- we want to use the OPPOSITE of goal image for the "what is the first move you'll take?" question
AUTOEVAL_IMAGE_PATHS = {
    "open_drawer": "assets/auto_eval_goal_images/close the drawer.png",
    "close_drawer": "assets/auto_eval_goal_images/open the drawer.png",
    "eggplant_to_basket": "assets/auto_eval_goal_images/put the eggplant in the blue sink.png",
    "eggplant_to_sink": "assets/auto_eval_goal_images/put the eggplant in the yellow basket.png",
}


async def run_test(
    model: str,
    env_desc: str,
    task_desc: str,
    image_path: str,
    num_samples: int = 5,
) -> list[dict[str, Any]]:
    """Run the test according to the given configuration."""
    vlm = VLM()
    request = create_vlm_request(model, image_path, env_desc, task_desc)
    responses = await vlm.generate_parallel([request] * num_samples)
    results = []

    for i, response, maybe_exception in responses:
        if maybe_exception or response is None:
            results.append({"index": i, "error": str(maybe_exception), "success": False})
        else:
            try:
                description, answer = parse_vlm_response(response.text)
                results.append(
                    {"index": i, "description": description, "answer": answer, "success": True}
                )
            except Exception as e:
                results.append(
                    {"index": i, "error": str(e), "raw_response": response.text, "success": False}
                )

    return results


def print_test_results(results: list[dict[str, Any]]) -> None:
    """Print the test results in a readable format."""
    for result in results:
        if result["success"]:
            print(f"Result {result['index']}:")
            print(result["description"])
            print("-")
            for k, v in result["answer"].items():
                print(f"{k}: {v}")
        else:
            print(f"Error in result {result['index']}: {result.get('error')}")
            if "raw_response" in result:
                print(f"Raw response: {result['raw_response']}")

        print("-" * 100)


async def main() -> None:
    """Main function to run the test."""
    # Set up the test configuration
    env_key = "sink"
    # env_key = "drawer"
    task_key = "eggplant_to_sink"
    # task_key = "eggplant_to_basket"
    # task_key = "close_drawer"
    num_samples = 2

    # Run the test
    results = await run_test(
        MODEL,
        AUTOEVAL_ENVIRONMENTS[env_key],
        AUTOEVAL_TASKS[task_key],
        AUTOEVAL_IMAGE_PATHS[task_key],
        num_samples=num_samples,
    )

    # Print the results
    print_test_results(results)


if __name__ == "__main__":
    asyncio.run(main())

# use https://github.com/zhouzypaul/mse-check
