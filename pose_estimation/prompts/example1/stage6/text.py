from api.azure_openai import local_image_to_data_url

assistant = """
# The pose is excellent now.
exit()
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
    {"role": "user", "content": "whether the ik is success: False"},
    {"role": "assistant", "content": assistant},
]