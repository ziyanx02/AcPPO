import argparse
import copy
import os
import pickle
import shutil

import numpy as np
import torch
import wandb
from envs.reward_wrapper import Backflip
from envs.locomotion_env import LocoEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


from scripts.locomotion.go2_backflip import main as backflip
from scripts.locomotion.go2_jump import main as jump
from scripts.locomotion.go2_walk import main as walk


MAIN_FUNCS = {
    'backflip': backflip,
    'jump': jump,
    'walk': walk,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_name', type=str, default='backflip')
    parser.add_argument('-t', '--task', type=str, default='backflip')
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=16384)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('-o', '--offline', action='store_true', default=False)
    parser.add_argument('--time', action='store_true', default=False)
    parser.add_argument('-p', '--ppo', action='store_true', default=False)

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--ckpt', type=int, default=1000)
    args = parser.parse_args()

    main = MAIN_FUNCS[args.task]
    main(args)


'''
# training
python train_backflip.py -e EXP_NAME

# evaluation
python eval_backflip.py -e EXP_NAME --ckpt NUM_CKPT
'''