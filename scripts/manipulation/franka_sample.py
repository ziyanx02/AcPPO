import argparse
import copy
import os
import pickle
import shutil
import yaml
from PIL import Image

import numpy as np
import torch
import wandb
from envs.manipulation_wrapper import PickCube
from envs.time_wrapper import TimeWrapper
from rsl_rl.runners import TDORunner

import genesis as gs


ENV_DICT = {
    'pickcube': PickCube,
}

import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from ddpm.ddpm import NoiseScheduler, MLP


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

    with open(f'./cfgs/franka_{args.task}.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    env_cfg = cfg['environment']
    env_cfg['substeps'] = 1
    env_cfg['control_freq'] = 1000

    env_class = ENV_DICT[args.task]
    env = env_class(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        show_viewer=args.vis,
        eval=False,
        debug=args.debug,
        n_render_envs=10,
        device=device,
    )

    state_min = env.init_state_min
    state_max = env.init_state_max
    contact_forces_limit = env.link_contact_forces_limit

    model = MLP(
        output_size=env.num_states,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        emb_size=args.emb_size,
        time_emb=args.time_emb,
        input_emb=args.input_emb,
    ).to(device)

    noise_scheduler = NoiseScheduler(
        num_timesteps=args.num_timesteps,
        beta_schedule=args.beta_schedule,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    num_envs = args.num_envs
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    def check_legal_state(env):
        contact_forces = torch.norm(env.link_contact_forces, dim=2)
        contact_mask = (contact_forces < contact_forces_limit).all(dim=1)
        ee_pos = env.ee_pos
        ee_pos_mask = torch.logical_and(ee_pos[:, 0] > 0, ee_pos[:, 2] < 0.4)
        mask = torch.logical_and(contact_mask, ee_pos_mask).float()
        return mask

    for epoch in range(num_epochs):
        model.train()
        samples = state_min + torch.rand(num_envs, env.num_states, device=device) * (state_max - state_min)
        actions = samples[:, :env.num_actions]
        env.set_state(samples, 0)
        env.step(actions)
        mask = check_legal_state(env)
        total_loss = 0
        for i in range(int(num_envs // batch_size)):
            batch = samples[i * batch_size: (i + 1) * batch_size]
            batch_mask = mask[i * batch_size: (i + 1) * batch_size]
            noise = torch.randn(batch.shape, device=device)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],), device=device,
            ).long()

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            loss = (noise_pred - noise) ** 2
            loss = loss.mean(dim=-1)
            loss *= batch_mask
            loss = loss.mean()
            loss.backward()
            total_loss += loss.detach().clone()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        # print(total_loss / (num_envs // batch_size + 0.01))
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            # generate data with the model to later visualize the learning process
            model.eval()
            sample = torch.randn(num_envs, env.num_states, device=device)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for t in timesteps:
                t = torch.from_numpy(np.repeat(t, num_envs)).long().to(device)
                with torch.no_grad():
                    residual = model(sample, t)
                sample = noise_scheduler.step(residual, t[0], sample)
            env.set_state(sample, 0)
            env.step(sample[:, :env.num_actions])
            mask = check_legal_state(env)
            print(f"Epoch {epoch} success rate:", (mask.sum() / mask.shape[0]).item())
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            frame_x, frame_y, frame_z = env.render_headless()
            image_x = Image.fromarray(frame_x)
            image_y = Image.fromarray(frame_y)
            image_z = Image.fromarray(frame_z)
            image_x.save(f"./logs/epoch_{epoch}_x.png")
            image_y.save(f"./logs/epoch_{epoch}_y.png")
            image_z.save(f"./logs/epoch_{epoch}_z.png")


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


'''
# training
python train_backflip.py -e EXP_NAME

# evaluation
python eval_backflip.py -e EXP_NAME --ckpt NUM_CKPT
'''