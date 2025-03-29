API_DEFINITION = '''
def rotate_quat_from_rpy(quat, roll, pitch, yaw):
    """
    Rotate a quaternion by given roll, pitch, and yaw angles (in degrees).
    
    Args:
        quat: Input quaternion as a tensor in shape (4,) with order (w, x, y, z)
        roll: Rotation around x-axis in degrees
        pitch: Rotation around y-axis in degrees
        yaw: Rotation around z-axis in degrees
        
    Returns:
        Rotated quaternion as a tensor in shape (4,)
    """

def get_body_link(self):
    """
    Get the body link.
    Return:
        link_id (int): The id of the link
    
    Body link is the link regarded as the base of the robot while simulating.
    Setting robot's position and quaternion is same as setting base link's position and quaternion.
    """
    return link_id

def set_body_link(self, link_id):
    """
    Set the body link.
    Args:
        link_id (int): The id of the link

    Body link is the link regarded as the base of the robot while simulating.
    Setting body link's position and quaternion is same as setting robot's position and quaternion.
    """

def get_body_pos(self):
    """
    Get the position of the body link.
    Return:
        body_pos (torch.tensor): The position of the body link
    """
    return body_pos

def set_body_pos(self, body_pos):
    """
    Set the position of the body link.
    Args:
        body_pos (torch.tensor): The target position of the body link
    """

def get_body_quat(self):
    """
    Get the quaternion of the body link.
    Return:
        body_quat (torch.tensor): The quaternion of the body link
    """
    return body_quat

def set_body_quat(self, body_quat):
    """
    Set the quaternion of the body link.
    Args:
        body_quat (torch.tensor): The target quaternion of the body link
    """

def get_link_pos(self, link_id=None):
    """
    Get the position of the link.
    Args:
        link_id (int): The id of the link
    Return:
        pos (torch.tensor): The position of the link
    """
    return pos

def get_link_quat(self, link_id=None):
    """
    Get the quaternion of the link.
    Args:
        link_id (int): The id of the link
    Return:
        quat (torch.tensor): The quaternion of the link
    """
    return quat

def set_link_pose(self, link_id, pos, quat=None):
    """
    Use inverse kinematics to set the position (and quaternion) of the link. If the quaternion is not provided, only the position is required.
    If the IK solver succeed, robot will be set to the pose; if not, the robot pose will not be changed.
    Args:
        link_id (int): The id of the link
        pos (torch.tensor): The target position of the link
        quat (torch.tensor): The target quaternion of the link
    Return:
        success (bool): Whether the IK solver succeeded
    """
    return success

def get_joints_between_links(self, link_id1, link_id2):
    """
    Get the joints between two links.
    Args:
        link_id1 (int): The id of the first link
        link_id2 (int): The id of the second link
    Return:
        joint_ids (list): The joints between the two links
    """
    return joint_ids

def get_joint_pos(self, joint_id):
    """
    Get the joint position.
    Args:
        joint_id (int): The id of the joint
    Return:
        joint_pos (float): The joint position
        joint_limit (tuple): The lower and upper bound of the joint position
    """
    return joint_pos, joint_limit

def set_joint_pos(self, joint_pos, joint_id):
    """
    Set the joint position.
    Args:
        joint_pos (float): The target joint position
        joint_id (int): The id of the joint
    """

def get_camera_pose(self):
    """
    Get the current camera pose.
    Return:
        camera_pose (dict): The current camera pose
        camera_pose["azimuth"] (degree): The azimuth angle of the camera
        camera_pose["elevation"] (degree): The elevation angle of the camera
        camera_pose["lookat"] (torch.tensor): The lookat point of the camera
        camera_pose["distance"] (float): The distance of the camera from the lookat point
    The camera's position is defined by the azimuth, elevation, lookat point, and distance by
    camera_pos = lookat + distance * torch.tensor([
        np.cos(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
        np.sin(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
        np.sin(elevation / 180 * np.pi),
    ])
    """
    return camera_pose

def set_camera_pose(self, camera_pose):
    """
    Set camera pose.
    Args:
        camera_pose (dict): The target camera pose
        camera_pose["azimuth"] (degree): The azimuth angle of the camera
        camera_pose["elevation"] (degree): The elevation angle of the camera
        camera_pose["lookat"] (torch.tensor): The lookat point of the camera
        camera_pose["distance"] (float): The distance of the camera from the lookat point
    The camera's position is defined by the azimuth, elevation, lookat point, and distance by
    camera_pos = lookat + distance * torch.tensor([
        np.cos(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
        np.sin(azimuth / 180 * np.pi) * np.cos(elevation / 180 * np.pi),
        np.sin(elevation / 180 * np.pi),
    ])
    """

def render(self):
    """
    Render current camera view
    """

def render_from_xyz(self):
    """
    Render three different views from x (front), y (left) and z (up)
    """

def update(self):
    """
    Update the robot display
    """

'''

API_EXAMPLES = f'''
You are given a robot with multiple links and joints. Here are the APIs that you can use to interact with the robot:
{API_DEFINITION}

You will be given a task about the robot. You should break down the task and complete subtasks one by one to get feedbacks after completing each subtask.

Here are some examples:
'''