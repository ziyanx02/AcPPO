from api.azure_openai import local_image_to_data_url

assistant = """
# To get a better understanding of the robot, I have to first look at the robot from different views
self.render_from_xyz()
messages = [
    ("image", "rgb_x", "rgb image from x"),
    ("image", "rgb_y", "rgb image from y"),
    ("image", "rgb_z", "rgb image from z"),
    ("image", "label_x", "labelled image from x"),
    ("image", "label_y", "labelled image from y"),
    ("image", "label_z", "labelled image from z"),
]
"""

prompt = [
    {"role": "user", "content": """
Your task is to generate a pose that the robot can walk with only its two front legs.
"""},
    {"role": "assistant", "content": assistant},
]