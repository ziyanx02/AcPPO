import re
import yaml

from reward_tuning.example import RESPONSE_SAMPLE_REWARD

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

def getRewardFactory(reward_function_code):
    exec_scope = {}
    exec(reward_function_code, exec_scope)
    if 'RewardWrapper' not in exec_scope:
        raise ValueError("RewardWrapper class was not defined in the provided code")
    return RewardFactory(exec_scope['RewardWrapper'])

def parse_response(response):
    match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
    if match : 
        reward_function_code = 'import torch\n\n' + match.group(1)
    else:
        raise ValueError("No reward function code found in response")

    match = re.search(r'```yaml\n(.*?)\n```', response, re.DOTALL) 
    if match :
        reward_scale_code = match.group(1)
    else:
        raise ValueError("No reward scale code found in response")
    
    reward_scales = yaml.safe_load(reward_scale_code)

    return reward_function_code, reward_scales['reward_scales']

if __name__ == '__main__':
    reward_function, reward_scales = parse_response(RESPONSE_SAMPLE_REWARD)
    factory = getRewardFactory(reward_function)

    gs.init(
        backend=gs.cpu,
        logging_level='warning',
    )

    cfg = yaml.safe_load(open('cfgs/go2-gait.yaml', 'r'))
    env_cfg = cfg['environment']

    new_class = factory.make(GaitEnv)
    
    env = new_class(1, env_cfg, False, False, True, device='cpu')
