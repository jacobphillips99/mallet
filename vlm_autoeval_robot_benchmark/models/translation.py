"""
Helpful components for translating between natural language and robot actions.
Targeted prompts to fall somewhere between AutoEval and ECoT
"""

import typing as t

# Robot description
ROBOT_DESCRIPTION = """
The robot has 7 degrees of freedom; imagine these from the point of view of the robot:
#0: X axis, meaning forward/backward, with forwards being towards the front (slightly right in the image) wall
#1: Y axis, meaning left/right, with left being towards the left wall
#2: Z axis, meaning up/down, with up being towards the ceiling and down being towards the floor or desk
#3: Tilt of the gripper (pitch)
#4: Roll of the gripper (roll)
#5: Rotation of the gripper (yaw)
#6: Gripper, meaning the gripper's opening/closing
""".strip()

# Method description
METHOD_DESCRIPTION = """
Your job is to output the action that will next help the robot complete the task.
This will just be the next timestep; you're not trying to accomplish the entire task, just what the robot should be doing in the next timestep.
""".strip()

HISTORY_INSTRUCTIONS = """
Consider the history provided in regards to the Historical Image as opposed to the provided Image.
Compare and contrast the historical state to the present state with respect to robot motion and the changes in the environment.
""".strip()


def get_output_format(gripper_position: t.Optional[str] = None, history: bool = False) -> str:
    """Generate the output format instructions."""
    return f"""
{HISTORY_INSTRUCTIONS if history else ""}
Describe the scene, including the robot, the task, the environment, and what progress the robot has made so far in completing the task.
Make sure to consider the 3D-position of the robot's gripper as well as the vantage point of the camera with respect to the scene.
Always consider whether the gripper is ACTUALLY grasping and closed around an object, not just whether it is in the vicinity of an object.
{f"For reference, the gripper position is currently {gripper_position}." if gripper_position else ""}
You may need to think about complex 3D relationships between the robot, the task, and the environment.
Next, write a paragraph about the action you'll need to take on this timestep, starting to think about how to break it down into primitive movements.

Once you've thought about the problem, start the JSON dictionary output. Remember to surround it with ```json tags.
Then, output a dictionary with 7 elements, each representing a degree of freedom of the robot.
The keys should be "x", "y", "z", "tilt", "roll", "rotation", and "gripper".
Choose the values from these options, according to the degrees of freedom described above.
Note that if the robot does not need to perform an action in a certain degree of freedom, just output null (we are using null instead of None in order to avoid JSON parsing issues)
{{
    "x": ["forward", "backward", null],
    "y": ["left", "right", null],
    "z": ["up", "down", null],
    "tilt": ["tilt up", "tilt down", null],
    "roll": ["roll up", "roll down", null],
    "rotation": ["rotate clockwise", "rotate counterclockwise", null],
    "gripper": ["open gripper", "close gripper"]
}}

In addition to each chosen direction value, also output float magnitude and a reason for why you chose that value.
For the magnitude, use a value between 0 and 1, where 0 is no movement and 1 is the maximum movement in the selected direction.
For the direction null, the magnitude should be 0, as no action will be taken.
The magnitude between 0 and 1 will be scaled to the range of the movement in the selected direction.

REMINDER! Note that for the gripper, there is no null option, as the gripper should always be open or closed.

The output should look like this example:

-------- BEGIN EXAMPLE --------
{"Description of the historical state and the changes with respect to the current state" if history else ""}
(insert description of scene and action here)
```json
{{
    "x": ["forward", 0.8, "I chose forward because the robot needs to move very far towards the front wall"],
    "y": ["left", 0.1,"I chose left because the robot needs to move a little bit towards the left wall in order to get closer to the drawer"],
    "z": [null, 0.0, "I chose null because the robot does not need to move up or down"],
    ...
    "gripper": ["close gripper", 1.0, "I chose closed because the robot needs to close the gripper to grasp the object"]
}}
```
-------- END EXAMPLE --------

Note: The output should NOT be all null actions unless the robot is done with the task.
""".strip()


def build_prompt(
    env_desc: str,
    task_desc: str,
    gripper_position: t.Optional[str] = None,
    history_flag: bool = False,
) -> str:
    """Build the prompt for the VLM."""
    return f"""
{env_desc}

{ROBOT_DESCRIPTION}

{task_desc}

{METHOD_DESCRIPTION}

{get_output_format(gripper_position, history_flag)}
"""
