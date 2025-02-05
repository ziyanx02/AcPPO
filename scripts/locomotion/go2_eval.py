import argparse
import copy
import os
import pickle

import numpy as np
import torch
from envs.reward_wrapper import Walk
from envs.reward_wrapper import Jump
from envs.reward_wrapper import Backflip
from envs.locomotion_env import LocoEnv
from envs.time_wrapper import TimeWrapper
from rsl_rl.runners import TDORunner

import genesis as gs


ENV_DICT = {
    'walk': Walk,
    'jump': Jump,
    'backflip': Backflip,
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

    env_class = ENV_DICT[args.task]
    env = env_class(
        num_envs=1,
        env_cfg=env_cfg,
        show_viewer=not args.headless,
        eval=False,
        debug=False,
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
            actions = policy(obs)
            env.step(actions)
            env.compute_observation()
            obs, _ = env.get_observations()
            n_frames += 1
            if args.record:
                if n_frames == 100:
                    env.stop_recording("backflip.mp4")
                    exit()

if __name__ == '__main__':
    main()