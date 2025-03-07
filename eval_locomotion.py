import argparse
import datetime
import copy
import os
import pickle

import numpy as np
import torch
from envs.locomotion_wrapper import Walk
from envs.locomotion_wrapper import Jump
from envs.locomotion_wrapper import Backflip
from envs.locomotion_wrapper import Walk_Gaits
from envs.locomotion_env import LocoEnv
from envs.time_wrapper import TimeWrapper
from rsl_rl.runners import TDORunner

import genesis as gs


ENV_DICT = {
    'walk': Walk,
    'jump': Jump,
    'backflip': Backflip,
    'gait': Walk_Gaits,
}

def export_policy_as_jit(actor_critic, path, name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f'{name}.pt')
    model = copy.deepcopy(actor_critic.actor).to('cpu')
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)

def main(args):

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    env_cfg, train_cfg = pickle.load(
        open(f'logs/{args.exp_name}/cfgs.pkl', 'rb')
    )
    env_cfg['reward']['reward_scales'] = {}
    env_cfg['PPO'] = True
    env_cfg['record_length'] = args.record_length
    if args.resample_time != None: env_cfg['resampling_time_s'] = args.resample_time

    env_class = ENV_DICT[args.task]
    env = env_class(
        num_envs=1,
        env_cfg=env_cfg,
        show_viewer=not args.headless,
        eval=True,
        debug=args.debug,
    )
    env = TimeWrapper(env, int(env_cfg['period_length_s'] * env_cfg['control_freq']), reset_each_period=False, observe_time=False)

    log_dir = f'logs/{args.exp_name}'

    runner = TDORunner(env, train_cfg, log_dir, device='cuda:0')

    resume_path = os.path.join(log_dir, f'model_{args.ckpt}.pt')
    runner.load(resume_path)
    path = os.path.join(log_dir, 'exported')
    export_policy_as_jit(runner.alg.actor_critic, path, args.exp_name + f'_ckpt{args.ckpt}')

    policy = runner.get_inference_policy(device='cuda:0')
    temporal_distribution = runner.get_inference_temporal_distribution(device='cuda:0')

    env.reset()
    env.compute_observation()
    obs, _ = env.get_observations()

    with torch.no_grad():
        stop = False
        n_frames = 0
        if args.record:
            env.start_recording(record_internal=False)
        while not stop:
            if args.td:
                state = temporal_distribution(env.time_buf)
                env.set_state(state)
                env.compute_observation()
                obs, _ = env.get_observations()
            actions = policy(obs)
            env.step(actions)
            env.compute_observation()
            obs, _ = env.get_observations()

            print(env.commands)
            n_frames += 1 
            if args.record and n_frames >= env.record_length: # 50 fps, 20 s
                env.stop_recording(args.exp_name + '.mp4')
                print('Finish recording!')
                break 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='go2_gait')
    parser.add_argument('-e', '--exp_name', type=str, default=None)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-r', '--record', action='store_true', default=False)
    parser.add_argument('-H', '--headless', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=1)
    parser.add_argument('--td', action='store_true', default=False)
    parser.add_argument('--ckpt', type=int, default=999)

    parser.add_argument('--record_length', help='unit: second', type=int, default=3)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--resample_time', help='unit: second', type=float, default=None)
    args = parser.parse_args()

    task_split = args.task.split('-')
    args.robot, args.task = task_split[0], task_split[1]

    main(args)


'''
# training
python train_locomotion.py -t go2-gait -e EXP_NAME

# evaluation
python eval_locomotion.py -t go2-gait -e EXP_NAME 
'''