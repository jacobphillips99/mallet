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

"""
We're going to focus on two `drawer` tasks: "open the drawer" and "close the drawer"
"""

"""
Looking at goal images, we might get an idea of if this is possible from the reset policy for the task we care about. 
E.g. We're focusing on "open the drawer", so the goal image of "close the drawer" shows the starting position of "open the drawer". At this point, we know
that we need the hand to go towards the left wall, meaning the Y component should be large and positive.
"""

import numpy as np
import asyncio
from pathlib import Path
import base64
import typing as t
import json
from PIL import Image 

from vlm_autoeval_robot_benchmark.utils.ecot_primitives.ecot_primitive_movements import describe_move, classify_movement
from vlm_autoeval_robot_benchmark.utils.ecot_primitives.inverse_ecot_primitive_movements import get_action_from_description
from vlm_autoeval_robot_benchmark.models.vlm import VLM, VLMRequest, ImageInput

MODEL = "gpt-4o"
vlm = VLM()
DELIMITER = "<answer>"

# goal_image_path= "assets/auto_eval_goal_images/close the drawer.png"
# goal_image_path = "assets/auto_eval_goal_images/open the drawer.png"
# goal_image_path = "assets/auto_eval_goal_images/put the eggplant in the blue sink.png"
goal_image_path = "assets/auto_eval_goal_images/put the eggplant in the yellow basket.png"
with open(goal_image_path, "rb") as f:
    image_data = f.read()

# environment = """
# You are looking at a wooden desk or table with a black robot arm on it.
# To the left of the robot arm is a drawer with a handle.
# The camera is slightly to the right of the robot. 
# """.strip()

environment = """
You are looking at a blue sink with a yellow basket on the left. 
There is an eggplant in the scene. 
The camera is slightly to the right of the robot.
""".strip() 

robot = """
The robot has 7 degrees of freedom:
#0: X axis, meaning forward/backward, with forwards being towards the front (slightly right in the image) wall
#1: Y axis, meaning left/right, with left being towards the left wall
#2: Z axis, meaning up/down, with up being towards the ceiling
#3: Tilt of the gripper (pitch)
#4: Roll of the gripper (roll)
#5: Rotation of the gripper (yaw)
#6: Gripper, meaning the gripper's opening/closing
""".strip()

# task = "The goal of the robot is to grab the handle of the drawer and open it."
# task = "The goal of the robot is to grab the handle of the drawer and close it."
# task = "The goal of the robot is to grab the eggplant and move it to the yellow basket."
task = "The goal of the robot is to grab the eggplant and move it to the blue sink."

method = """
Your job is to output the action that will help the robot complete the task. 
This will just be the next timestep; you're not trying to accomplish the entire task, just what the robot should be doing in the next second or so. 
""".strip()

output_format = f"""
First, describe the scene.
Next, extensively describe the action you'll need to take on this timestep, starting to think about how to break it down into primitive movements.
Once you've thought about the problem, output a {DELIMITER} tag and then start the dictionary output (no need to do ```json or anything like that)
Then, output a dictionary with 7 elements, each representing a degree of freedom of the robot.
The keys should be "x", "y", "z", "tilt", "roll", "rotation", and "gripper".
Choose the values from these options, according to the degrees of freedom described above.
Note that if the robot does not need to perform an action in a certain degree of freedom, just output None.
{{
    "x": ["forward", "backward", "None"],
    "y": ["left", "right", "None"],
    "z": ["up", "down", "None"],
    "tilt": ["tilt up", "tilt down", "None"],
    "roll": ["roll up", "roll down", "None"],
    "rotation": ["rotate clockwise", "rotate counterclockwise", "None"],
    "gripper": ["open", "close", "None"]
}}

In addition to each chosen value, also output a reason for why you chose that value.
The output should look like this example: 

```
(insert description of scene and action here)
{DELIMITER}
{{
    "x": ["backward", "I chose backward because the robot needs to move towards the back / its own body"],
    "y": ["left", "I chose left because the robot needs to move towards the left wall in order to get closer to the drawer"],
    ...
    "gripper": ["None", "I chose None because there is no need for the robot to take any movement with the gripper right now as we are not close to the drawer"]
}}
```

Below is an image of the scene.
""".strip()

PROMPT = f"""
{environment}

{robot}

{task}

{method}

{output_format}
"""

async def main():
    request = VLMRequest(
        model=MODEL,
        prompt=PROMPT,
        images=[
            ImageInput(
                data=base64.b64encode(image_data).decode('utf-8'),
                mime_type="image/png"
            )
        ],
    )
    responses = await vlm.generate_parallel([request] * 5)
    try:
        for i, response, maybe_exception in responses:
            if maybe_exception:
                print(f"Error: {maybe_exception}")
            else:
                description, answer = response.text.split(DELIMITER)
                print(description)
                answer = json.loads(answer)
                for k, v in answer.items():
                    print(f"{k}: {v}")
                print("-" * 100)
    except Exception as e:
        print(f"Error: {e}")
        print(response.text)
    breakpoint()
    print(response)

if __name__ == "__main__":
    asyncio.run(main())

# use https://github.com/zhouzypaul/mse-check