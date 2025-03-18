import os
import pickle
import yaml
import shutil
import torch

import genesis as gs
from envs.locomotion_wrapper import GaitEnv
from envs.time_wrapper import TimeWrapper
from reward_tuning.parse import getRewardFactory, parse_response
from rsl_rl.runners import TDORunner


def train(return_queue, args, response, iter_id, sample_id, train_cfg, env_cfg, tune_cfg):
    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )

    reward_function, reward_scales = parse_response(response)
    RewardFactory = getRewardFactory(reward_function)
    env_cfg['reward']['reward_scales'] = reward_scales
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
    num_logpoints = tune_cfg['num_logpoints']
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

    # Return can't be tensor due to multiprocess
    for i in log_dict.keys():
        for key in log_dict[i].keys():
            if type(log_dict[i][key]) == torch.Tensor:
                log_dict[i][key] = log_dict[i][key].item()


    return_queue.put({
        'iter_id': iter_id,
        'sample_id': sample_id,
        'exp_name': exp_name,
        'response': {
            'raw': response,
            'reward_scales': reward_scales,
            'reward_function': reward_function,
        },
        'train_log': log_dict,
        'log_frequency': log_period,
        'max_episode_length': env.max_episode_length, 
    })