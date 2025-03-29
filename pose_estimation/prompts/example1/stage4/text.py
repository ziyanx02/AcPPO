from api.azure_openai import local_image_to_data_url

assistant = """
# Now the left foot is on the back of the center of the robot, so I have to move it a litle bit forward
left_foot_pos = self.get_link_pos(left_foot_id)
left_foot_pos[0] += 0.05
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
    {"role": "user", "content": [
                                    {"type": "text", "text": "current camera view"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": local_image_to_data_url("./rgb.png")},
                                    },
                                ],},
    {"role": "user", "content": [
                                    {"type": "text", "text": "rgb image from x"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": local_image_to_data_url("./rgb_x.png")},
                                    },
                                ],},
    {"role": "user", "content": [
                                    {"type": "text", "text": "rgb image from y"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": local_image_to_data_url("./rgb_y.png")},
                                    },
                                ],},
    {"role": "user", "content": [
                                    {"type": "text", "text": "rgb image from z"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": local_image_to_data_url("./rgb_z.png")},
                                    },
                                ],},
    {"role": "user", "content": [
                                    {"type": "text", "text": "current camera view with labels"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": local_image_to_data_url("./label.png")},
                                    },
                                ],},
    {"role": "user", "content": [
                                    {"type": "text", "text": "labelled image from x"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": local_image_to_data_url("./label_x.png")},
                                    },
                                ],},
    {"role": "user", "content": [
                                    {"type": "text", "text": "labelled image from y"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": local_image_to_data_url("./label_y.png")},
                                    },
                                ],},
    {"role": "user", "content": [
                                    {"type": "text", "text": "labelled image from z"},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": local_image_to_data_url("./label_z.png")},
                                    },
                                ],},
    {"role": "user", "content": "whether the ik is success: True"},
    {"role": "assistant", "content": assistant},
]