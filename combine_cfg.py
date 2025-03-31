import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='panda')
parser.add_argument('--cfg_train', type=str, default='panda-gait-hop')
parser.add_argument('--cfg_pose', type=str, default='basic')
parser.add_argument('--cfg_save', type=str, default=None)
args = parser.parse_args()

cfg_train = yaml.safe_load(open(f"./cfgs/{args.cfg_train}.yaml"))
cfg_pose = yaml.safe_load(open(f"./pose_estimation/cfgs/{args.robot}/{args.cfg_pose}.yaml"))

cfg = cfg_train
env_cfg = cfg['environment']

env_cfg['urdf_path'] = cfg_pose['robot']['asset_path']
env_cfg['robot_scale'] = cfg_pose['control']['robot_scale']
env_cfg['links_to_keep'] = cfg_pose['robot']['links_to_keep']
num_dof = len(cfg_pose['control']['default_joint_angles'])
env_cfg['num_actions'] = num_dof
env_cfg['num_dofs'] = num_dof
env_cfg['num_states'] = 10 + num_dof * 2

terminate = []
for link in cfg_pose['robot']['link_names']:
    if link not in cfg_pose['robot']['foot_names']:
        terminate.append(link)
# import pdb; pdb.set_trace()
env_cfg['termination_contact_link_names'] = terminate.copy()
env_cfg['penalized_contact_link_names'] = terminate.copy()
env_cfg['feet_link_names'] = cfg_pose['robot']['foot_names']
env_cfg['base_link_name'] = [cfg_pose['robot']['body_name']]
env_cfg['PD_stiffness'] = {'': cfg_pose['control']['kp']}
env_cfg['PD_damping'] = {'': cfg_pose['control']['kd']}
env_cfg['armature'] = cfg_pose['control']['armature']
env_cfg['dof_damping'] = cfg_pose['control']['damping'] 
env_cfg['dof_names'] = list(cfg_pose['control']['default_joint_angles'].keys())
env_cfg['base_init_pos'] = cfg_pose['control']['base_init_pos']
env_cfg['base_init_quat'] = cfg_pose['control']['base_init_quat']
env_cfg['body_init_pos'] = cfg_pose['control']['body_init_pos']
env_cfg['body_init_quat'] = cfg_pose['control']['body_init_quat']
env_cfg['default_joint_angles'] = cfg_pose['control']['default_joint_angles']

num_feet = len(cfg_pose['robot']['foot_names'])
base_height = cfg_pose['control']['body_init_pos'][2] * 0.9
env_cfg['gait']['base_height_target'] = base_height
env_cfg['gait']['frequency'] = [3.0,] * num_feet
env_cfg['gait']['duration'] = [0.5,] * num_feet
env_cfg['gait']['offset'] = [0.5 * (i % 2) for i in range(num_feet)]
env_cfg['gait']['feet_height_target'] = [0.4 * base_height,] * num_feet
env_cfg['gait']['stationary_position'] = cfg_pose['control']['stationary_position']

env_cfg['observation']['num_obs'] = 9 + num_dof * 3 + num_feet
env_cfg['observation']['num_priv_obs'] = 12 + num_dof * 4 + num_feet

env_cfg['command']['lin_vel_x_range'] = [0., cfg_pose['control']['diameter']]
env_cfg['command']['lin_vel_y_range'] = [0., 0.]
env_cfg['command']['ang_vel_range'] = [0., 0.]

cfg['reward_tuning'] = {
    'num_iterations': 3,
    'num_samples': 4,
    'num_logpoints': 10,
}

if args.cfg_save != None:
    yaml.safe_dump(cfg, 
                   open(f"./cfgs/{args.cfg_save}.yaml", "w"), 
                   sort_keys=False,
                   default_flow_style=False)