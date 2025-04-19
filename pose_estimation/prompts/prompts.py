ORIENTATION_PROMPT = {
    "x": """
This image shows the robot looking from the front.
The blue arrow shows the z-axis (upward). The green arrow shows the y-axis (leftward).
**Important**:
- 'Up' in the image is upward for the robot.
- 'Left' in the image is the robot's right side.
- 'Right' in the image is the robot's left side.
""",
    "y": """
This image shows the robot looking from its left side.
The blue arrow shows the z-axis (upward). The red arrow shows the x-axis (forward).
**Important**:
- 'Up' in the image is upward for the robot.
- 'Left' in the image is the robot's front.
- 'Right' in the image is the robot's back.
""",
    "z": """
This image shows the robot looking from above.
The red arrow shows the x-axis (forward). The green arrow shows the y-axis (leftward).
**Important**:
- 'Up' in the image is the robot's front.
- 'Down' in the image is the robot's back.
- 'Left' in the image is the robot's left side.
- 'Right' in the image is the robot's right side.
""",
    "-x": """
This image shows the robot looking from the back.
The blue arrow shows the z-axis (upward). The green arrow shows the y-axis (leftward).
**Important**:
- 'Up' in the image is upward for the robot.
- 'Left' in the image is the robot's left side.
- 'Right' in the image is the robot's right side.
""",
    "-y": """
This image shows the robot looking from its right side.
The blue arrow shows the z-axis (upward). The red arrow shows the x-axis (forward).
**Important**:
- 'Up' in the image is upward for the robot.
- 'Left' in the image is the robot's back.
- 'Reft' in the image is the robot's front.
""",
    "-z": """
This image shows the robot looking from bottom.
The blue arrow shows the z-axis (upward). The green arrow shows the y-axis (leftward).
**Important**:
- 'Up' in the image is the robot's front.
- 'Down' in the image is the robot's back.
- 'Left' in the image is the robot's right side.
- 'Right' in the image is the robot's left side.
""",
}
SEGMENTATION_PROMPT = "The image shows the robot with its parts (called links) highlighted in different colors. Each link is labeled with a unique number."
MOVEMENT_LABEL_PROMPT = "The image shows the possible movements of the robot's link."
SYSTEM_PROMPT = "You are an expert in robot kinematics and motion planning."
BODY_SELECTION_PROMPT = """
The robot in the images is divided into multiple parts (called links). Each link is highlighted with a unique color and labeled with an ID.

Visible link IDs in the images:
{visible_links_id}

The robot's target walking behavior is described as:
{task}

Your task:
- Identify the base link. The base is the part of the robot that should stay stable while the robot moves according to velocity commands.

Instructions:
- Respond with **only one number** — the ID of the base link.
"""
ROTATION_PROPOSE_PROMPT = """
You are helping to rotate a robot to complete this task: "{task}"
The rotation follows right-hand rule.
Rotation around the x-axis goes from +y to +z, from +z to -y.
Rotation around the y-axis goes from +z to +x, from +x to -z.
Rotation around the z-axis goes from +x to +y, from +y to -x.

- The base of the robot is link {body_link_id}, and its rotation should remain fixed during completing the task.
- Propose how the robot should be rotated to better match the target task.

Respond in this format:

<your analysis>
Answer:
yes
<axis>
<angle>

or

<your analysis>
Answer:
no

<your analysis> is the analysis of the robot's current pose and how it can be improved.
<axis> can be either **x**, **y**, or **z**
<angle> should be specified in **degrees**.
"""
ROTATION_EVALUATE_PROMPT = """
You are helping to rotate a robot to complete this task: "{task}"

The robot has just been rotated around axis '{axis}' by {angle} degrees.

Based on the images before and after the rotation, decide:
- Should we **keep this rotation** and consider further rotation?
- Should we **cancel the last rotation**?
- Or is the **robot well-aligned now**?

Format:

<your analysis>
Answer:
continue

or

<your analysis>
Answer:
cancel

or

<your analysis>
Answer:
done
"""
MOVEMENT_SELECTION_PROMPT = """
Your task is to adjust the robot pose for a standing pose to complete the task: {task}
Now the link {body_link_id} is **adjusted and fixed**. So **don't** consider the rotation of link {body_link_id}.

You are moving link {link_id} by inverse kinametics (as link {body_link_id} is fixed so only joint between link {body_link_id} and link {link_id} will move during ik).
The potential positions are labeled by ids in the annotated images, including: {sampled_points}

First analysis the front / back, left / right in each image.
Then analysis what usage is link {link_id} for. For example, if link {link_id} needs to touch the ground, then it should be put lower.
For each potential positions (ids), analysis why or why not you are going to select it.

At the end of your reply, you should contain the ids you think are reasonable in this format (remamber to add \\n after Answer:):
<Your analysis>
Answer:
<ids list>

<Your analysis> is the analysis you made.
<ids list> should be the ids (numbers) you selected, seperate by a comma.

The blue arrow is z axis (upward). The red arrow is x axis (forward). The gree arrow is y axis (leftward).
"""
POSE_COMPARISON_PROMPT = """
Your task is to find a standing pose of the robot to complete the task: {task}

You will be given two set of pictures. One is before an adjustment, one is after. You are going to decide whether to keep this adjustment.
This is not the final adjustment, so you only need to consider whether **this adjustment** make the robot pose **better**.
If the adjustment make the pose better or there is no difference, return "Continue".
If the adjustment make the pose worse, return "Cancel".

This is the format you should follow in the reply:
<Your analysis>
Answer:
<Your decision>

<Your analysis> is the analysis you made.
<Your decision> should be either Continue or Cancel

The blue arrow is z axis (upward). The red arrow is x axis (forward). The gree arrow is y axis (leftward).
"""
NUMERIC_ADJUSTMENT_PROMPT = """
Your task is to adjust the robot pose to better complete the task: {task}

The robot link {body_link_id} is fixed. You should move exrtemities {extremities} to fit the task and comply symmetry and some commensense (i.e. the height of feet should be the same, when there are only two feet then the line should across center of mass).
The current positions of the extremities are (the order is same as previous extremites' order):
{extremities_pos_as_prompt}

The current center of mass is {com}.

First you have to analysis what rule you should comply. Then from the current extremety positions, find out reasonable target positions.

At the end of your reply, you should contain the target position in this format:
<Your analysis>
Answer:
<extremety 1 position>
<extremety 2 position>
...

<Your analysis> should contain your analysis about extremities' position and com.
<extremety i position> should be a list containing x, y and z positions. The order and format of extremities' position should be same as the current positions of the extremities provided.
"""
NUMERIC_COMPARISON_PRMPT = """
Do you think further adjustment is needed? Please respond with one of: yes / no
If yes, please provide the new target positions in the same format as before.
Example 1:
no

Example 2:
yes
Reason why the previous answer is not good.
Answer:
[0.0, -0.2, -0.4]
[0.0, 0.2, -0.4]
[0.2, 0.2, -0.0]
[-0.15, -0.2, -0.0]
"""