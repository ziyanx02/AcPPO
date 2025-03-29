from api.azure_openai import local_image_to_data_url

assistant = """
# The inverse kinematics failed, perhaps due to the leg is not long enough. So I should also move the foot a little bit higher.
left_foot_pos = self.get_link_pos(left_foot_id)
left_foot_pos[0] -= 0.1
left_foot_pos[2] += 0.1
ik_success = self.set_link_pose(left_foot_id, left_foot_pos)

if ik_success:
    # After adjusting the link position, I should step
    self.update()

    # I should check the robot pose after update.
    self.render()
    self.render_from_xyz()
    messages = [
        ("image", "rgb", "current camera view"),
        ("image", "rgb_x", "rgb image from x"),
        ("image", "rgb_y", "rgb image from y"),
        ("image", "rgb_z", "rgb image from z"),
        ("image", "label", "current camera view with labels"),
        ("image", "label_x", "labelled image from x"),
        ("image", "label_y", "labelled image from y"),
        ("image", "label_z", "labelled image from z"),
        ("bool", ik_success, "whether the ik is success")
    ]
else:
    # The inverse kinematics failed, so the link is not moved. Now I have to try a new way.
    messages = [
        ("bool", ik_success, "whether the ik is success")
    ]
"""

prompt = [
    {"role": "user", "content": "whether the ik is success: False"},
    {"role": "assistant", "content": assistant},
]