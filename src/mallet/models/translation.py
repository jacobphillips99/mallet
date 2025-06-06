"""
Modular components for translating between natural language and robot actions.
Provides building blocks for creating structured prompts for VLM interactions.
"""

import typing as t

from pydantic import BaseModel

# Core descriptions
ROBOT_DESCRIPTION = """
The robot has 7 degrees of freedom, described from the robot's perspective.
Imagine a clock face with the workspace in the center:
- The robot is positioned at 6 o'clock
- The camera is at 5 o'clock, pointing inward at the workspace
- The robot and camera are both facing toward the center of the workspace

#0: X axis - Forward/backward movement (forward is 12 o'clock, backward is 6 o'clock. This can also mean forward towards the front (slightly right) wall, or backwards towards the robot body.)
#1: Y axis - Left/right movement (left is 9 o'clock, right is 3 o'clock. This can also mean left towards the left wall, or right towards the camera.)
#2: Z axis - Up/down movement (up is above the clock, down is below the clock. This can also mean up towards the ceiling, or down towards the floor.)
#3: Pitch - Tilt of the gripper (up/down angle)
#4: Roll - Roll of the gripper (rotation around forward axis)
#5: Yaw - Rotation of the gripper (left/right angle)
#6: Gripper - Opening/closing of the gripper
""".strip()

METHOD_DESCRIPTION = """
Your task is to determine the robot's next action to help complete the assigned task.
Focus only on the immediate next step, not the entire task sequence.
""".strip()

HISTORY_INSTRUCTIONS = """
You will also be provided with historical image(s) and actions. Consider these when determining the next action.
Compare the historical image(s) with the current image to identify changes in:
- Robot position and orientation
- Environmental changes
- Progress toward the task goal
This should help you determine the impact of prior actions and adjust your plan accordingly.
You need to critically consider the impact of each historical (action, image) tuple on the environment.
If the historical action is incorrect, you need to reverse course or adjust your plan based on the impact of previous actions.
""".strip()

SCENE_DESCRIPTION_INSTRUCTIONS = """
Provide a clear description of the current scene including:
- Robot's position and orientation
- Gripper state (open/closed) and whether it's actually grasping an object
- Relevant objects in the environment
- Progress toward completing the task
- Camera perspective relative to the scene
- 2D bounding box of the robot's gripper and any objects related to the task
Consider all 3D spatial relationships between the robot, objects, and environment.
""".strip()

ACTION_PLANNING_INSTRUCTIONS = """
Explain your reasoning for the next action, considering:
- Current robot state and position
- Spatial relationship to target objects
- Required movements to make progress
- Constraints or obstacles to avoid
""".strip()

# Output format components
JSON_FORMAT_INSTRUCTIONS = """
First, describe the scene according to the image, following the instructions in the Scene Description section.
{historical_format_instructions}
Make sure to focus on the relationship between the robot's gripper and the task objects.
Next, explain your reasoning for the next action, following the instructions in the Action Planning section.

Finally, output your answer as a Python JSON dictionary, beginning with ```json and ending with ```.
Include 7 elements, each representing a degree of freedom:
{{
    "x": ["forward", "backward", null],
    "y": ["left", "right", null],
    "z": ["up", "down", null],
    "tilt": ["tilt up", "tilt down", null],
    "roll": ["roll up", "roll down", null],
    "rotation": ["rotate clockwise", "rotate counterclockwise", null],
    "gripper": ["open gripper", "close gripper"]
}}

For each direction:
1. Select the appropriate direction from the options listed (or null if no movement)
2. Include a magnitude between 0.0 and 1.0 (0 = no movement, 1 = maximum movement)
3. Provide a brief explanation for your choice

IMPORTANT: The gripper must always be set to either "open gripper" or "close gripper" (never null).
""".strip()

EXTRA_HINTS = """
Play close attention to these extra hints!
- Always consider the z-axis that describes the height of the robot's gripper, especially when grasping, lifting, placing, or avoiding obstacles.
- Always keep the gripper open until the robot is ready to grasp an object.
- Always consider the 3D orientation of objects in the scene, especially the robot's gripper.
- Think about parallax and perspective -- is the gripper actually at the same location as the object, or does it just appear that way?
- Always double check if the gripper is ACTUALLY grasping an object. If so, the bounding box of the gripper should be very similar to the bounding box of the object.
- It's okay for the gripper to open or close while moving the rest of the robot. Don't delay moving the end-effector in xyz space just to open or close the gripper, unless you're actually grasping an object.
""".strip()

OUTPUT_EXAMPLE = """
-------- BEGIN EXAMPLE --------
**Scene Description:**
(explanation of the scene according to the image and history and action here)

**Action Planning:**
(explanation of the reasoning for the next action here)
{historical_output_example}
**Output:**
```json
{{
    "x": ["forward", 0.8, "Need to reach the drawer handle which is in front of the robot"],
    "y": ["left", 0.1, "Need to align slightly left to center on the drawer handle"],
    "z": [null, 0.0, "No vertical adjustment needed as gripper is already at handle height"],
    "tilt": [null, 0.0, "Current gripper angle is appropriate for grasping the handle"],
    "roll": [null, 0.0, "No roll adjustment needed for this grasping position"],
    "rotation": [null, 0.0, "Current rotation is aligned with the drawer handle"],
    "gripper": ["close gripper", 1.0, "Need to firmly grasp the drawer handle before pulling"]
}}
```
-------- END EXAMPLE --------

Note: The output should NOT contain all null actions unless the robot has completed the task.
""".strip()

HISTORY_PREFIX = "You are being shown the history of a robotics episode."
HISTORY_SUFFIX = "Consider the above history when determining the next action.\n"


class PromptTemplate(BaseModel):
    """
    Prompts must fill in the following fields
    """

    env_desc: str
    task_desc: str
    gripper_position: t.Optional[str] = None
    history_flag: bool = False


class PromptSegment:
    """A segment of a prompt with defined content and name."""

    def __init__(self, content: str, name: str = ""):
        """Initialize a prompt segment.

        Args:
            content: The text content of the segment
            name: A name for the segment for easier reference
        """
        self.content = content
        self.name = name

    def __str__(self) -> str:
        return self.content


class PromptBuilder:
    """Builder class for constructing modular prompts."""

    def __init__(self) -> None:
        """Initialize an empty prompt builder."""
        self.segments: list[PromptSegment] = []

    def add_segment(self, segment: PromptSegment) -> "PromptBuilder":
        """Add a segment to the prompt.

        Args:
            segment: The prompt segment to add

        Returns:
            Self for chaining
        """
        self.segments.append(segment)
        return self

    def add_content(self, content: str, name: str = "") -> "PromptBuilder":
        """Add content as a new segment.

        Args:
            content: The text content to add
            name: Optional name for the segment

        Returns:
            Self for chaining
        """
        segment = PromptSegment(content, name=name)
        return self.add_segment(segment)

    def add_gripper_position(self, gripper_position: str) -> "PromptBuilder":
        """Add gripper position information to the prompt.

        Args:
            gripper_position: The gripper position string

        Returns:
            Self for chaining
        """
        content = f"For reference, the gripper position is currently {gripper_position}."
        return self.add_content(content, name="gripper_position")

    def add_custom_instructions(
        self, instructions: str, name: str = "custom_instructions"
    ) -> "PromptBuilder":
        """Add custom instructions to the prompt.

        Args:
            instructions: The custom instructions text
            name: Optional name for the segment

        Returns:
            Self for chaining
        """
        return self.add_content(instructions, name=name)

    def build(self) -> str:
        """Build the final prompt by concatenating all segments.

        Returns:
            The complete prompt string
        """
        return "\n\n".join(
            f"# {segment.name.replace('_', ' ').title() if segment.name is not None else ''}\n{segment.content}"
            for segment in self.segments
        )


def build_standard_prompt(
    prompt_template: PromptTemplate,
) -> str:
    """Build a standard prompt for the VLM.
    Args:
        prompt_template: A PromptTemplate object containing environment description,
            task description, gripper position, and history flag

    Returns:
        The constructed prompt
    """
    builder = PromptBuilder()

    # Add core components
    builder.add_content(METHOD_DESCRIPTION, name="method_description")
    builder.add_content(ROBOT_DESCRIPTION, name="robot_description")
    builder.add_content(prompt_template.env_desc, name="environment_description")
    builder.add_content(prompt_template.task_desc, name="task_description")

    # Add scene and action instructions
    builder.add_content(SCENE_DESCRIPTION_INSTRUCTIONS, name="scene_description_instructions")

    if prompt_template.gripper_position:
        builder.add_gripper_position(prompt_template.gripper_position)
    builder.add_content(ACTION_PLANNING_INSTRUCTIONS, name="action_planning_instructions")

    # Add extra hints
    builder.add_content(EXTRA_HINTS, name="extra_hints")

    # Conditionally add history instructions
    if prompt_template.history_flag:
        builder.add_content(HISTORY_INSTRUCTIONS, name="history_instructions")

    # Add output format instructions
    historical_format_instructions = (
        ""
        if not prompt_template.history_flag
        else "Next, describe the progress between the historical image and the current image as according to the History Instructions section."
    )
    builder.add_content(
        JSON_FORMAT_INSTRUCTIONS.format(
            historical_format_instructions=historical_format_instructions
        ),
        name="output_format_instructions",
    )
    historical_output_example = (
        ""
        if not prompt_template.history_flag
        else "**Historical Change**\n(explanation of the historical change here)"
    )
    builder.add_content(
        OUTPUT_EXAMPLE.format(historical_output_example=historical_output_example),
        name="output_example",
    )
    return builder.build()
