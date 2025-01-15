import torch
import torch.nn as nn
import torch.optim as optim

episode_length = 100
num_envs = 1000
num_dims = 5

class NumberEnv:
    def __init__(self, num_envs, num_dims, device="cuda"):
        self.device = device

        self.num_envs = num_envs
        self.num_actions = num_dims
        self.num_observations = num_dims
        self.num_states = num_dims

        # Initialize environment states and time steps
        self.states = torch.zeros((num_envs, num_dims), device=self.device)

    def set_state(self, states):
        self.states = states

    def get_state(self):
        return self.states

    def check_termination(self):
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def compute_observation(self):
        return self.states
    
    def compute_reward(self):
        return torch.zeros(self.num_envs, device=self.device)

    def step(self, action):
        self.states = self.states + action

        return self.compute_observation(), self.compute_reward(), self.check_termination()

class VectorEnv:
    def __init__(self, env: NumberEnv, episode_length, device="cuda"):
        self.env = env
        self.num_envs = env.num_envs
        self.num_states = env.num_states
        self.episode_length = episode_length
        self.device = device
        self.terminate_after_each_episode = True

        # Initialize environment states and time steps
        self.time_steps = torch.arange(self.num_envs, dtype=torch.int64, device=self.device).remainder(self.episode_length)

        # Parameters for inducing state distribution
        self.mean_params = nn.Parameter(torch.zeros((episode_length, self.num_states), device=self.device))
        self.std_params = nn.Parameter(torch.ones((episode_length, self.num_states), device=self.device))

        self.optimizer = optim.Adam([self.mean_params, self.std_params], lr=0.03)

    def sample_initial_state(self, num_samples):
        initial_state = torch.zeros((num_samples, self.num_states), device=self.device)
        return initial_state

    def reset_idx(self, idx):
        means = self.mean_params[self.time_steps[idx]]
        stds = self.std_params[self.time_steps[idx]]
        states = self.env.get_state()
        states[idx] = torch.distributions.Normal(means, stds).sample()
        states[torch.logical_and(idx, self.time_steps == 0)] = self.sample_initial_state(torch.logical_and(idx, self.time_steps == 0).sum())
        self.env.set_state(states)

    def reset(self):
        self.time_steps = torch.arange(self.num_envs, dtype=torch.int64, device=self.device).remainder(self.episode_length)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def step(self, action):

        value_obs, reward, terminated = self.env.step(action)

        # Update states
        self.time_steps = (self.time_steps + 1) % self.episode_length
        if self.terminate_after_each_episode:
            terminated = torch.logical_or(terminated, self.time_steps == 0)
        self.reset_idx(terminated)

        actor_obs, states = self.env.compute_observation(), self.env.get_state()

        # Log probabilities of updated states
        means = self.mean_params[self.time_steps]
        stds = self.std_params[self.time_steps]
        dist = torch.distributions.Normal(means, stds)
        log_prob = dist.log_prob(states).sum(dim=-1)  # Sum over dimensions

        return value_obs, actor_obs, reward, terminated, log_prob

# Example usage
env = NumberEnv(num_envs, num_dims)
vecenv = VectorEnv(env, episode_length)
vecenv.reset()

for epoch in range(200):
    log_probs = []
    # vecenv.reset()
    for _ in range(10):
        action = 0.1 + 0.1 * torch.randn(num_envs, num_dims, device="cuda")
        value_obs, actor_obs, reward, terminated, log_prob = vecenv.step(action)
        log_probs.append(log_prob)
    log_probs = torch.cat(log_probs)
    loss = -log_probs.mean()

    vecenv.optimizer.zero_grad()
    loss.backward()
    vecenv.optimizer.step()

    vecenv.std_params.data.clamp_(min=0.01)

    print(f"Epoch {epoch}: Loss {loss.item()}")

print(vecenv.mean_params.data)
print(vecenv.std_params.data)