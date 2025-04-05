"""
Helpful components for translating between natural language and robot actions.
Targeted prompts to fall somewhere between AutoEval and ECoT
"""

DELIMITER = "<answer>"

# Robot description
ROBOT_DESCRIPTION = """
The robot has 7 degrees of freedom; imagine these from the point of view of the robot:
#0: X axis, meaning forward/backward, with forwards being towards the front (slightly right in the image) wall
#1: Y axis, meaning left/right, with left being towards the left wall
#2: Z axis, meaning up/down, with up being towards the ceiling
#3: Tilt of the gripper (pitch)
#4: Roll of the gripper (roll)
#5: Rotation of the gripper (yaw)
#6: Gripper, meaning the gripper's opening/closing
""".strip()

# Method description
METHOD_DESCRIPTION = """
Your job is to output the action that will help the robot complete the task.
This will just be the next timestep; you're not trying to accomplish the entire task, just what the robot should be doing in the next second or so.
""".strip()


def get_output_format(delimiter: str) -> str:
    """Generate the output format instructions."""
    return f"""
First, describe the scene, including the robot, the task, the environment, and what progress the robot has made so far in completing the task.
Next, write a paragraph about the action you'll need to take on this timestep, starting to think about how to break it down into primitive movements.
Once you've thought about the problem, output a {delimiter} tag and then start the dictionary output (no need to do ```json or anything like that)
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
{delimiter}
{{
    "x": ["backward", "I chose backward because the robot needs to move towards the back / its own body"],
    "y": ["left", "I chose left because the robot needs to move towards the left wall in order to get closer to the drawer"],
    ...
    "gripper": ["None", "I chose None because there is no need for the robot to take any movement with the gripper right now as we are not close to the drawer"]
}}
```

Below is the image of the scene:
""".strip()


def build_prompt(env_desc: str, task_desc: str) -> str:
    """Build the prompt for the VLM."""
    return f"""
{env_desc}

{ROBOT_DESCRIPTION}

{task_desc}

{METHOD_DESCRIPTION}

{get_output_format(DELIMITER)}
"""
