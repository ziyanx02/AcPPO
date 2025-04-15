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
- 'Left' in the image is the robot's front.
- 'Reft' in the image is the robot's back.
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

You are given images of the robot from different views.
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
Your task is find a standing pose of the robot to complete the task: {task}
Now the link {body_link_id} is adjusted and fixed. So the rotation of the robot is not your task now.
You are moving link {link_id} by inverse kinametics (as link {body_link_id} is fixed so only joint between link {body_link_id} and link {link_id} will change during ik).
The potential positions are labeled by ids in the annotated images, including: {sampled_points}

First analysis the front / back, left / right in each image.
Then analysis which way you should move the link. And for the image taken from each axis, find out reasonable movements. Do not consider the connections between different views.

At the end of your reply, you should contain the ids you think are reasonable in this format (remamber to add \\n after Answer:):
Example1:
Answer:
x:
1, 3, 9, 14
y:
2, 3
z:
3, 5, 7

Example2:
Answer:
x:
3, 7
y:
2, 7, 9, 15
z:
5, 8

The blue arrow is z axis (upward). The red arrow is x axis (forwward). The gree arrow is y axis (leftward).
Here are the images:
"""