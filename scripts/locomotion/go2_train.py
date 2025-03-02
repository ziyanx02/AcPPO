import argparse
import copy
import os
import pickle
import shutil
import yaml

import numpy as np
import torch
import wandb
from envs.locomotion_wrapper import Walk
from envs.locomotion_wrapper import Jump
from envs.locomotion_wrapper import Backflip
from envs.time_wrapper import TimeWrapper
from rsl_rl.runners import TDORunner

import genesis as gs


ENV_DICT = {
    'walk': Walk,
    'jump': Jump,
    'backflip': Backflip,
}


def main(args):

    if args.debug or args.eval:
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
    with open(f'./cfgs/go2_{args.task}.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    train_cfg = cfg['learning']
    env_cfg = cfg['environment']

    if args.debug:
        train_cfg['record_interval'] = -1
    if not args.offline:
        train_cfg['logger'] = 'wandb'
        train_cfg['exp_name'] = args.exp_name
        train_cfg['print_infos'] = False
    if args.ppo:
        train_cfg['PPO'] = True
        env_cfg['PPO'] = True

    if args.eval:
        env_cfg['episode_length_s'] = 1 / env_cfg['control_freq']

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, train_cfg],
        open(f'{log_dir}/cfgs.pkl', 'wb'),
    )

    env_class = ENV_DICT[args.task]
    env = env_class(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        show_viewer=args.vis,
        eval=False,
        debug=args.debug,
        device=device,
    )

    env = TimeWrapper(env, int(env_cfg['period_length_s'] * env_cfg['control_freq']), reset_each_period=False, observe_time=args.time)

    runner = TDORunner(env, train_cfg, log_dir, device=device)

    if args.resume is not None:
        resume_dir = f'logs/{args.resume}'
        resume_path = os.path.join(resume_dir, f'model_{args.ckpt}.pt')
        print('==> resume training from', resume_path)
        runner.load(resume_path)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', type=str, default=None)
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=10000)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('-o', '--offline', action='store_true', default=False)
    parser.add_argument('-p', '--ppo', action='store_true', default=False)
    parser.add_argument('--time', action='store_true', default=False)
    parser.add_argument('-t', '--task', type=str, default='jump')

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--ckpt', type=int, default=1000)
    args = parser.parse_args()
    main(args)