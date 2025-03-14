import argparse
import datetime
import os
import pickle
import shutil
import yaml
import copy

import torch.multiprocessing as mp
import numpy as np
import torch
import wandb
from envs.locomotion_wrapper import GaitEnv
from envs.time_wrapper import TimeWrapper
from rsl_rl.runners import TDORunner

import os
from reward_tuning.prompt import INITIAL_USER, INITIAL_SYSTEM
from reward_tuning.client import Client
from reward_tuning.parse import parse_response

import genesis as gs


def train(response, iter_id, sample_id, train_cfg, env_cfg, args, num_logpoints):

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )

    RewardFactory, reward_function, reward_scale = parse_response(response)
    env_cfg['reward']['reward_scales'] = reward_scale
    env_cfg['reward']['reward_function'] = reward_function
    exp_name = f'{args.exp_name}_it{iter_id}_{sample_id}'
    device = 'cpu' if args.cpu else 'cuda'
    log_dir = f'logs/{exp_name}'
    if not args.offline:
        train_cfg['logger'] = 'wandb'
        train_cfg['exp_name'] = exp_name
        train_cfg['print_infos'] = False
    if args.ppo:
        train_cfg['PPO'] = True
        env_cfg['PPO'] = True
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump([env_cfg, train_cfg], open(f'{log_dir}/cfgs.pkl', 'wb'),)
    yaml.dump([env_cfg, train_cfg], open(f'{log_dir}/cfgs.yaml', 'w'),)
    print(reward_function, file=open(f'{log_dir}/reward_wrapper.py', 'w'))


    max_iterations = args.max_iterations
    log_dict = {}
    log_period = max_iterations // num_logpoints
    for i in range(num_logpoints):
        log_dict[max_iterations - log_period * (num_logpoints - i - 1) - 1] = {}

    env = RewardFactory.make(GaitEnv)(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        show_viewer=args.vis,
        eval=False,
        debug=args.debug,
        device=device,
    )

    env = TimeWrapper(env, int(env_cfg['period_length_s'] * env_cfg['control_freq']), reset_each_period=False, observe_time=args.time)
    runner = TDORunner(env, train_cfg, log_dir, device=device, log_dict=log_dict)
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)

    return log_dict

def main(args):    
    mp.set_start_method("spawn")

    client = Client(disable=args.disable)
    base_message = [
        {"role": "system", "content": INITIAL_SYSTEM},
        {"role": "user", "content": INITIAL_USER}
    ]

    with open(f'./cfgs/{args.cfg}', 'r') as file:
        cfg = yaml.safe_load(file)
    train_cfg = cfg['learning']
    env_cfg = cfg['environment']
    tune_cfg = cfg['reward_tuning']

    for iter_id in range(tune_cfg['num_iterations']):

        for sample_id in range(tune_cfg['num_samples']):
            # try:
            response = client.response(base_message)
            training_process = mp.Process(
                target=train,
                args=(response, iter_id, sample_id, copy.deepcopy(train_cfg), copy.deepcopy(env_cfg), args, tune_cfg['num_logpoints'])
            )
            training_process.start()

            # except Exception as e:
            #     print(f"Iteration {iter_id}_{sample_id} Error: {str(e)}")
            #     print('Waiting for debugging...')
            #     import pdb; pdb.set_trace()
            #     continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='go2-gait')
    parser.add_argument('-e', '--exp_name', type=str, default=None)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=15000)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('-o', '--offline', action='store_true', default=False)
    parser.add_argument('--time', action='store_true', default=False)
    parser.add_argument('-p', '--ppo', action='store_false', default=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--disable', action='store_true', default=False)
    args = parser.parse_args()

    if args.exp_name == None:
        args.exp_name = args.task
        now = datetime.datetime.now()
        args.exp_name += f'_{now.month}-{now.day}-{now.hour}-{now.minute}'

    args.cfg = args.task + '.yaml'
    args.robot = args.task.split('-')[0]
    
    if args.debug:
        args.vis = True
        args.offline = True
        args.num_envs = 1
        args.cpu = True

    if not torch.cuda.is_available():
        args.cpu = True

    main(args)
