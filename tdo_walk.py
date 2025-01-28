import argparse
import copy
import os
import pickle
import shutil

import numpy as np
import torch
import wandb
from envs.reward_wrapper import Go2
from envs.time_wrapper import TimeWrapper
from rsl_rl.runners import TDORunner

import genesis as gs


def get_train_cfg(args):

    train_cfg_dict = {
        'algorithm': {
            'clip_param': 0.2,
            'desired_kl': 0.01,
            'entropy_coef': 0.003,
            'gamma': 0.99,
            'lam': 0.95,
            'learning_rate': 0.001,
            'max_grad_norm': 1.0,
            'num_learning_epochs': 5,
            'num_mini_batches': 4,
            'schedule': 'adaptive',
            'use_clipped_value_loss': True,
            'value_loss_coef': 1.0,
            'class_name': 'TDO',
        },
        'temporal_distribution': {
            'learning_rate': 0.001,
            'class_name': 'TemporalDistribution',
        },
        'policy': {
            "class_name": "ActorCritic",
            'activation': 'elu',
            'actor_hidden_dims': [512, 256, 128],
            'critic_hidden_dims': [512, 256, 128],
            'init_noise_std': 1.0,
        },
        'save_interval': 100,
        'runner_class_name': 'OnPolicyRunner',
        'num_steps_per_env': 24,
        'reset_rate': 0.0,
        'seed': 1,
        # "logger": "wandb",
        'empirical_normalization': False,
        'wandb_project': 'TDO',
        'wandb_entity': 'ziyanx02',
        'exp_name': None,
        'record_interval': 50,
        'print_infos': True,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        'urdf_path': 'urdf/go2/urdf/go2.urdf',
        'links_to_keep': ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot',],
        'num_actions': 12,
        'num_dofs': 12,
        'num_states': 33, # z, pitch, roll, lin vel, ang vel, dof pos, dof vel,
        # joint/link names
        'default_joint_angles': {  # [rad]
            'FL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,
            'RL_hip_joint': 0.0,
            'RR_hip_joint': 0.0,
            'FL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        },
        'dof_names': [
            'FR_hip_joint',
            'FR_thigh_joint',
            'FR_calf_joint',
            'FL_hip_joint',
            'FL_thigh_joint',
            'FL_calf_joint',
            'RR_hip_joint',
            'RR_thigh_joint',
            'RR_calf_joint',
            'RL_hip_joint',
            'RL_thigh_joint',
            'RL_calf_joint',
        ],
        'termination_contact_link_names': ['base'],
        'penalized_contact_link_names': ['base', 'thigh', 'calf'],
        'feet_link_names': ['foot'],
        'base_link_name': ['base'],
        # PD
        'PD_stiffness': {'joint': 40.0},
        'PD_damping': {'joint': 2.0},
        # termination
        'reset_after_termination': False,
        'termination_if_roll_greater_than': 0.4,
        'termination_if_pitch_greater_than': 0.4,
        'termination_if_height_lower_than': 0.0,
        # base pose
        'base_init_pos': [0.0, 0.0, 0.42],
        'base_init_quat': [1.0, 0.0, 0.0, 0.0],
        'pos_randomization': 0.0,
        'rot_randomization': 0.0,
        'dof_pos_randomization': 0.3,
        # random push
        'push_interval_s': -1,
        'max_push_vel_xy': 1.0,
        # time (second)
        'episode_length_s': 20.0,
        'use_timeout': True,
        'period_length_s': 0.5,
        'resampling_time_s': 4.0,
        'command_type': 'ang_vel_yaw',  # 'ang_vel_yaw' or 'heading'
        'action_scale': 0.25,
        'delay_action': True,
        'clip_actions': 100.0,
        'control_freq': 50,
        'decimation': 4,
        'feet_geom_offset': 1,
        'use_terrain': False,
        # domain randomization
        'dof_damping': 0.0,
        'armature': 0.05,
        'randomize_friction': True,
        'friction_range': [0.2, 1.5],
        'randomize_base_mass': True,
        'added_mass_range': [-1., 3.],
        'randomize_com_displacement': True,
        'com_displacement_range': [-0.01, 0.01],
        'randomize_motor_strength': False,
        'motor_strength_range': [0.9, 1.1],
        'randomize_motor_offset': True,
        'motor_offset_range': [-0.02, 0.02],
        'randomize_kp_scale': True,
        'kp_scale_range': [0.8, 1.2],
        'randomize_kd_scale': True,
        'kd_scale_range': [0.8, 1.2],
    }
    obs_cfg = {
        'use_time_indicator': False,
        'num_obs': 9 + 3 * env_cfg['num_dofs'],
        'num_history_obs': 1,
        'obs_noise': {
            'ang_vel': 0.1,
            'gravity': 0.02,
            'dof_pos': 0.01,
            'dof_vel': 0.5,
        },
        'obs_scales': {
            'lin_vel': 1.0,
            'ang_vel': 1.0,
            'dof_pos': 1.0,
            'dof_vel': 0.05,
        },
        'num_priv_obs': 12 + 4 * env_cfg['num_dofs'],
    }
    reward_cfg = {
        'tracking_sigma': 0.25,
        'soft_dof_pos_limit': 0.9,
        'base_height_target': 0.3,
        'reward_scales': {
            'tracking_lin_vel': 1.0,
            'tracking_ang_vel': 0.5,
            'lin_vel_z': -2.0,
            'ang_vel_xy': -0.05,
            'orientation': -1.,
            'base_height': -5.,
            'torques': -0.0001,
            'collision': -1.,
            'dof_vel': -0.,
            'dof_acc': -2.5e-7,
            'feet_air_time': 1.0,
            'action_rate': -0.01,
            'dof_pos_diff': -0.3,
            'terminate': -10.0,
        },
    }
    command_cfg = {
        'lin_vel_x_range': [-1.0, 1.0],
        'lin_vel_y_range': [-1.0, 1.0],
        'ang_vel_range': [-2.0, 2.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', type=str, default='Go2')
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=10000)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('-o', '--offline', action='store_true', default=False)

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--ckpt', type=int, default=1000)
    args = parser.parse_args()

    if args.debug:
        args.vis = True
        args.offline = True
        args.num_envs = 1
        args.cpu = True

    if not torch.cuda.is_available():
        args.cpu = True

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )
    device = 'cpu' if args.cpu else 'cuda'

    log_dir = f'logs/{args.exp_name}'
    train_cfg = get_train_cfg(args)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    if args.debug:
        train_cfg['record_interval'] = -1
    if not args.offline:
        train_cfg['logger'] = 'wandb'
        train_cfg['exp_name'] = args.exp_name
        train_cfg['print_infos'] = False

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = Go2(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
        eval=args.eval,
        debug=args.debug,
        device=device,
    )
    env = TimeWrapper(env, int(env_cfg['period_length_s'] * env_cfg['control_freq']), reset_each_period=False)

    runner = TDORunner(env, train_cfg, log_dir, device=device)

    if args.resume is not None:
        resume_dir = f'logs/{args.resume}'
        resume_path = os.path.join(resume_dir, f'model_{args.ckpt}.pt')
        print('==> resume training from', resume_path)
        runner.load(resume_path)

    # wandb.login(key='1d5fe5b941feff91e5dbb834d4f687fdbec8e516')
    # wandb.init(project='genesis', name=args.exp_name, entity='ziyanx02', dir=log_dir, mode='offline' if args.offline else 'online', config=train_cfg)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg],
        open(f'{log_dir}/cfgs.pkl', 'wb'),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    main()


'''
# training
python train_backflip.py -e EXP_NAME

# evaluation
python eval_backflip.py -e EXP_NAME --ckpt NUM_CKPT
'''