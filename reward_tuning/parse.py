import re
import yaml

from reward_tuning.example import RESPONSE_SAMPLE

import numpy as np
import torch
import torch.nn as nn

import genesis as gs

from envs.locomotion_wrapper import GaitEnv

class RewardFactory:
    def __init__(self, RewardWrapper):
        self.RewardWrapper = RewardWrapper
    def make(self, base_class):
        return type(f"{base_class.__name__}Reward", (base_class, self.RewardWrapper), {})

def parse_response(response):
    match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    if match : 
        reward_function_code = 'import torch\n\n' + match.group(1)
    else:
        raise ValueError("No reward function code found in response")

    # Define a dictionary to act as the local scope for exec
    exec_scope = {}

    # Execute the reward function code in the custom scope
    exec(reward_function_code, exec_scope)

    if 'RewardWrapper' not in exec_scope:
        raise ValueError("RewardWrapper class was not defined in the provided code")

    match = re.search(r'```yaml\n(.*?)\n```', response, re.DOTALL) 
    if match :
        reward_scale_code = match.group(1)
    else:
        raise ValueError("No reward scale code found in response")
    
    reward_scale = yaml.safe_load(reward_scale_code)

    return RewardFactory(exec_scope['RewardWrapper']), reward_function_code, reward_scale['reward_scales']

if __name__ == '__main__':
    factory, _, _ = parse_response(RESPONSE_SAMPLE)

    gs.init(
        backend=gs.cpu,
        logging_level='warning',
    )

    cfg = yaml.safe_load(open('cfgs/go2-gait.yaml', 'r'))
    env_cfg = cfg['environment']

    new_class = factory.make(GaitEnv)
    
    env = new_class(1, env_cfg, False, False, True, device='cpu')
