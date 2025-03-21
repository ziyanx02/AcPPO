import os
import pickle
import yaml
import shutil
import traceback

import torch
import genesis as gs
from envs.metric_wrapper import GaitEnvMetric
from envs.time_wrapper import TimeWrapper
from rsl_rl.runners import TDORunner



def eval(return_queue, args, exp_name):

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        logging_level='warning',
    )

    device = 'cpu' if args.cpu else 'cuda'
    log_dir = f'logs/{exp_name}'
    env_cfg, train_cfg = pickle.load(
        open(f'logs/{exp_name}/cfgs.pkl', 'rb')
    )
    env_cfg['reward']['reward_scales'] = {}
    env_cfg['PPO'] = True
    env_cfg['record_length'] = args.record_length
    env_cfg['resampling_time_s'] = args.resample_time

    env = GaitEnvMetric(
        num_envs=1,
        env_cfg=env_cfg,
        show_viewer=args.vis,
        eval=True,
        debug=True,
    )
    env = TimeWrapper(env, int(env_cfg['period_length_s'] * env_cfg['control_freq']), reset_each_period=False, observe_time=False)

    log_dir = f'logs/{exp_name}'

    runner = TDORunner(env, train_cfg, log_dir, device=device)

    resume_path = os.path.join(log_dir, f'model_{args.max_iterations - 1}.pt')
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    env.reset()
    env.compute_observation()
    obs, _ = env.get_observations()

    metric = {}
    max_n_frames = args.num_eval_step
    if args.record:
        max_n_frames = max(max_n_frames, env.record_length)

    with torch.no_grad():
        stop = False
        n_frames = 0
        record = args.record 
        if record:
            env.start_recording(record_internal=False)
        while n_frames < max_n_frames:
            actions = policy(obs)
            env.step(actions)
            env.compute_observation()
            obs, extras = env.get_observations()

            n_frames += 1 

            for key in extras['metric'].keys():
                if key not in metric.keys():
                    metric[key] = 0
                metric[key] += extras['metric'][key]
                
            if record and n_frames >= env.record_length: # 50 fps, 20 s
                record = False
                env.stop_recording(f'{log_dir}/{args.exp_name}.mp4')
                print('Finish recording!')
            
        for key in metric.keys():
            if key != 'terminate':
                metric[key] /= max_n_frames
    
    for key in metric.keys():
        if type(metric[key]) == torch.Tensor:
            metric[key] = metric[key].item()

    result = {'metric': metric}
    pickle.dump(result, open(f'{log_dir}/result_eval.pkl', 'wb'),)
    return_queue.put(result)

def eval_try(return_queue, *args):
    try:
        eval(return_queue, *args)
    except Exception:
        return_queue.put({
            'error': traceback.format_exc()
        })