import numpy as np
import torch
import math

from rsl_rl.env import VecEnv
import matplotlib.pyplot as plt

class GridEnv(VecEnv):
    def __init__(
        self,
        num_envs,
        env_cfg,
        device='cuda',
    ) -> None:

        self.cfg = env_cfg
        self.reward_cfg = env_cfg['reward']
        self.reward_scales = self.reward_cfg['reward_scales']

        self.num_envs = num_envs
        self.num_actions = 2
        self.num_states = 2
        self.device = device
        self.half_grid_size = 5
        self.grid_size = 2 * self.half_grid_size

        self.rew_buf = torch.zeros(
            (num_envs,), device=device, dtype=torch.float
        )
        self.reset_buf = torch.ones(
            (num_envs,), device=device, dtype=torch.bool
        )
        self.state_buf = torch.zeros(
            (num_envs, 2), device=device, dtype=torch.float
        )
        self.actions = torch.zeros(
            (num_envs, 2), device=device, dtype=torch.float
        )
        self.in_grid = torch.zeros(
            (num_envs,), device=device, dtype=torch.int
        )
        self.state_min = -self.half_grid_size * torch.ones(
            (num_envs, 2), device=device, dtype=torch.float
        )
        self.state_max = self.half_grid_size * torch.ones(
            (num_envs, 2), device=device, dtype=torch.float
        )
        self.dis_to_target = torch.zeros(
            (num_envs,), device=device, dtype=torch.float
        )
        self.last_dis_to_target = torch.zeros(
            (num_envs,), device=device, dtype=torch.float
        )
        self.state_min[:, 0] = self.in_grid * self.grid_size - self.half_grid_size
        self.state_max[:, 0] = self.in_grid * self.grid_size + self.half_grid_size
        self.episode_length_buf = torch.zeros(
            (num_envs,), device=device, dtype=torch.int
        )
        self.max_episode_length = 20
        self.gate_pos = torch.tensor([self.half_grid_size, 3], device=device, dtype=torch.float)
        self.gate_width = 0.2
        self.target_pos = torch.tensor([self.grid_size, 0], device=device, dtype=torch.float)
        self.target_width = 0.5
        self.extras = {}

        self._prepare_reward_function()
        self._prepare_temporal_distribution()

    def _prepare_reward_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == 'termination':
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
            for name in self.reward_scales.keys()
        }

    def _prepare_temporal_distribution(self):
        init_state_mean = torch.zeros(
            (2,), device=self.device, dtype=torch.float,
        )
        init_state_std = torch.zeros(
            (2,), device=self.device, dtype=torch.float,
        )
        self.state_mean = init_state_mean.unsqueeze(0).repeat(self.max_episode_length, 1)
        self.state_mean[:, 0] = torch.arange(self.max_episode_length, device=self.device, dtype=torch.float) / self.max_episode_length * self.grid_size
        self.state_std = init_state_std.unsqueeze(0).repeat(self.max_episode_length, 1)
        self.init_state_min = torch.tensor(
            [-self.half_grid_size, -self.half_grid_size], device=self.device, dtype=torch.float
        )
        self.init_state_max = torch.tensor(
            [self.grid_size + self.half_grid_size, self.half_grid_size], device=self.device, dtype=torch.float
        )

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return None, None

    def step(self, actions):

        self.actions = actions
        new_state = self.state_buf + actions
        new_state = torch.max(new_state, self.state_min)
        new_state = torch.min(new_state, self.state_max)
        self.state_buf = new_state
        self.extras['critic_obs'] = self.state_buf.clone()

        dis_to_gate = torch.norm(self.state_buf - self.gate_pos, dim=1)
        self.in_grid[dis_to_gate < self.gate_width] += 1
        self.in_grid = self.in_grid.clip(0, 1)

        self.last_dis_to_target = self.dis_to_target.clone()
        self.dis_to_target = torch.norm(self.state_buf - self.target_pos, dim=1)

        self.state_min[:, 0] = self.in_grid * self.grid_size - self.half_grid_size
        self.state_max[:, 0] = self.in_grid * self.grid_size + self.half_grid_size

        self.episode_length_buf += 1
        self.reset_buf = self.episode_length_buf >= self.max_episode_length
        envs_idx = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(envs_idx)
        self.compute_reward()
        self.extras['time_outs'] = self.reset_buf

        return (
            self.state_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        init_states = torch.zeros(
            (len(envs_idx), 2), device=self.device, dtype=torch.float
        )
        self.set_state(init_states, 0, envs_idx)

    def compute_reward(self):
        self.rew_buf[:] = 0.
        self.extras['rewards'] = {}
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            rew = torch.clip(rew, -100.0, 100.0)
            self.rew_buf += rew
            self.episode_sums[name] += rew
            self.extras['rewards'][name] = torch.mean(rew).item()

    def get_state(self):
        return self.state_buf, self.episode_length_buf

    def get_observations(self):
        return self.state_buf, self.extras

    def get_privileged_observations(self):
        return self.state_buf

    def set_state(self, states, times, envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self.device)
        if len(envs_idx) == 0:
            return

        self.episode_length_buf[envs_idx] = times
        self._set_state(states, envs_idx)

    def _set_state(self, states, envs_idx):
        self.state_buf[envs_idx] = states
        self.dis_to_target[envs_idx] = torch.norm(self.state_buf[envs_idx] - self.target_pos, dim=1)
        self.last_dis_to_target[envs_idx] = self.dis_to_target[envs_idx].clone()
        self.in_grid[envs_idx] = (self.state_buf[envs_idx, 0] > self.half_grid_size).int()
        self.reset_buf[envs_idx] = 1

    def resample_commands(self, envs_idx):
        pass

    def compute_observation(self):
        pass

    def compute_critic_observation(self):
        pass

    def render(self, resolution=0.05):

        # Initialize a 2D histogram to count the number of states in each bin
        hist = torch.zeros((math.floor((2 * self.grid_size) / resolution) + 2, math.floor(self.grid_size / resolution) + 1), device=self.device)

        # Iterate over each environment's state and count the number of states in each bin
        states = self.state_buf.clone()
        states[:, 0] += self.in_grid * resolution
        idx = torch.floor((states + self.half_grid_size) / resolution).to(torch.int64)
        for i in range(self.num_envs):
            hist[idx[i, 0], idx[i, 1]] += 1

        return hist

    def show_heatmap(self, hist, log=True):

        # Convert the histogram to a numpy array for plotting
        if isinstance(hist, torch.Tensor):
            hist = hist.cpu().numpy()

        # Convert the histogram to a log scale for better visualization
        if log:
            hist = np.log(hist + 1)  # Add 1 to avoid log(0)

        # Plot the heatmap
        plt.figure(figsize=(16, 8))
        plt.imshow(hist.T, origin='lower', extent=[-self.half_grid_size, self.grid_size + self.half_grid_size, -self.half_grid_size, self.half_grid_size], cmap='viridis')
        plt.colorbar(label='Log Count')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Log Number of States in Each Grid Cell')
        plt.show()

    def save_heatmap(self, hist, filename, log=True):

        # Convert the histogram to a numpy array for plotting
        if isinstance(hist, torch.Tensor):
            hist = hist.cpu().numpy()

        # Convert the histogram to a log scale for better visualization
        if log:
            hist = np.log(hist + 1)  # Add 1 to avoid log(0)

        # Plot the heatmap
        plt.figure(figsize=(16, 8))
        plt.imshow(hist.T, origin='lower', extent=[-self.half_grid_size, self.grid_size + self.half_grid_size, -self.half_grid_size, self.half_grid_size], cmap='viridis')
        plt.colorbar(label='Log Count')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Log Number of States in Each Grid Cell')
        plt.savefig(filename)
        plt.close()