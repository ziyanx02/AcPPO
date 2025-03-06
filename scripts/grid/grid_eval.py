import argparse
import copy
import os
import pickle

import math
import numpy as np
import torch
import wandb
from envs.grid_wrapper import TwoGrids
from envs.time_wrapper import TimeWrapper
from rsl_rl.runners import TDORunner

ENV_DICT = {
    'suc_noreg': TwoGrids,
    'dis_noreg': TwoGrids,
    'suc_reg': TwoGrids,
    'dis_reg': TwoGrids,
}


def export_policy_as_jit(actor_critic, path, name):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, f'{name}.pt')
    model = copy.deepcopy(actor_critic.actor).to('cpu')
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)

def main(args):

    device = 'cpu' if args.cpu else 'cuda'
    args.num_envs = 2000

    env_cfg, train_cfg = pickle.load(
        open(f'logs/{args.exp_name}/cfgs.pkl', 'rb')
    )
    env_cfg['reward']['reward_scales'] = {}

    env_class = ENV_DICT[args.task]
    env = env_class(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        device=device,
    )
    env = TimeWrapper(env, 20, reset_each_period=False, observe_time=False)

    log_dir = f'logs/{args.exp_name}'

    runner = TDORunner(env, train_cfg, log_dir, device='cuda:0')

    for i in range(11):
        args.ckpt = i * 10

        resume_path = os.path.join(log_dir, f'model_{args.ckpt}.pt')
        runner.load(resume_path)
        # path = os.path.join(log_dir, 'exported')
        # export_policy_as_jit(runner.alg.actor_critic, path, args.exp_name + f'_ckpt{args.ckpt}')

        occupency_metric = 0
        resolution = 0.2
        with torch.no_grad():

            # vis eval occupancy metric
            env.reset()
            obs, _ = env.get_observations()
            for _ in range(20):
                obs, _ = env.get_observations()
                actions = runner.alg.actor_critic.act(obs, env.episode_length_buf)
                env.step(actions)
                obs, _ = env.get_observations()
                hist = env.render(resolution=resolution)
                occupency_metric += hist
            env.save_heatmap(occupency_metric, os.path.join(log_dir, f'om_eval_{args.ckpt}.png'))

            # vis eval occupancy metric
            mean = runner.alg.temporal_distribution.mean_params.data.repeat(100, 1)
            std = runner.alg.temporal_distribution.std_params.data.repeat(100, 1)
            states = mean + std * torch.randn_like(std)
            env.set_state(states)
            obs, _ = env.get_observations()
            for _ in range(20):
                obs, _ = env.get_observations()
                actions = runner.alg.actor_critic.act(obs, env.episode_length_buf)
                env.step(actions)
                obs, _ = env.get_observations()
                hist = env.render(resolution=resolution)
                occupency_metric += hist
            env.save_heatmap(occupency_metric, os.path.join(log_dir, f'om_train_{args.ckpt}.png'))

            # vis td
            mean = runner.alg.temporal_distribution.mean_params.data.repeat(100, 1)
            std = runner.alg.temporal_distribution.std_params.data.repeat(100, 1)
            hist = torch.zeros((math.floor((2 * env.grid_size) / resolution) + 2, math.floor(env.grid_size / resolution) + 1), device=runner.device)
            states = mean + std * torch.randn_like(std)
            idx = torch.floor((states + env.half_grid_size) / resolution).to(torch.int64)
            idx[:, 0] = torch.clip(idx[:, 0], min=0, max=int(2 * env.grid_size / resolution))
            idx[:, 1] = torch.clip(idx[:, 1], min=0, max=int(env.grid_size / resolution))
            for i in range(idx.shape[0]):
                hist[idx[i, 0], idx[i, 1]] += 1
            env.save_heatmap(hist, os.path.join(log_dir, f'td_{args.ckpt}.png'))

            # vis critic
            x = torch.arange(-env.half_grid_size, env.grid_size + env.half_grid_size, resolution)  # x coordinates
            y = torch.arange(-env.half_grid_size, env.half_grid_size, resolution)  # x coordinates
            xx, yy = torch.meshgrid(x, y, indexing='xy')
            grid = torch.stack([xx, yy], dim=-1).swapaxes(0, 1)
            grid_shape = grid.shape[:-1]
            grid = grid.reshape(-1, 2)
            values = runner.alg.actor_critic.critic(grid.to(runner.device)).detach().cpu()
            values = values.reshape(*grid_shape).numpy()
            env.save_heatmap(values, os.path.join(log_dir, f'value_{args.ckpt}.png'), log=False)

if __name__ == '__main__':
    main()