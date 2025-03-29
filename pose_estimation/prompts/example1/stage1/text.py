from api.azure_openai import local_image_to_data_url

assistant = """
# link 0 is in the center of the robot which is good for a body link
self.set_body_link(0)

# The camera should look at the body link when adjusting the body link
camera_pose = self.get_camera_pose()
body_pos = self.get_body_pos()
camera_pose["lookat"] = body_pos
self.set_camera_pose(camera_pose)

# The task asks to generate a pose that the robot can walk with only its two front legs, so the rear legs should be lifted.
# We'll add a pitch rotation -45 degrees
body_quat = self.get_body_quat()
body_quat = rotate_quat_from_rpy(body_quat, 0, -45, 0)
self.set_body_quat(body_quat)

# After adjusting the body link and robot pose, I should step
self.update()

# I should check the robot pose after update
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
]
"""

prompt = [
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
    {"role": "assistant", "content": assistant},
]
