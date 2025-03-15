import argparse
import datetime
import yaml

import torch.multiprocessing as mp
import torch
from reward_tuning.prompt import INITIAL_USER, INITIAL_SYSTEM
from reward_tuning.client import Client

from scripts.train import train
from scripts.eval import eval

def train_eval(return_queue, args, response, iter_id, sample_id, train_cfg, env_cfg, tune_cfg):
    train_queue = mp.Queue()
    eval_queue = mp.Queue()

    train_process = mp.Process(
        target=train,
        args=(train_queue, args, response, iter_id, sample_id, train_cfg, env_cfg, tune_cfg),
    )
    train_process.start()
    train_process.join()
    train_return = train_queue.get()

    eval_process = mp.Process(
        target=eval,
        args=(eval_queue, args, train_return['exp_name'])
    )
    eval_process.start()
    eval_process.join()
    eval_return = eval_queue.get()

    return_queue.put({
        'train': train_return,
        'eval': eval_return,
    })


def main(args):    
    mp.set_start_method("spawn", force=True)

    client = Client(disable=args.disable)
    base_message = [
        {"role": "system", "content": INITIAL_SYSTEM},
        {"role": "user", "content": INITIAL_USER}
    ]

    with open(f'./cfgs/{args.cfg}', 'r') as file:
        cfg = yaml.safe_load(file)
    train_cfg = cfg['learning']
    env_cfg = cfg['environment']
    tune_cfg = cfg['reward_tuning']

    for iter_id in range(tune_cfg['num_iterations']):
        return_queue = mp.Queue()
        process = []

        for sample_id in range(tune_cfg['num_samples']):
            # try:
            response = client.response(base_message)
            sample_process = mp.Process(
                target=train_eval,
                args=(return_queue, args, response, iter_id, sample_id, train_cfg, env_cfg, tune_cfg)
            )
            sample_process.start()
            process.append(sample_process)

            # except Exception as e:
            #     print(f"Iteration {iter_id}_{sample_id} Error: {str(e)}")
            #     print('Waiting for debugging...')
            #     import pdb; pdb.set_trace()
            #     continue
        

        error_process = []
        for sample_id, p in enumerate(process): 
            p.join()
            if p.exitcode != 0:
                error_process.append(sample_id)
        print(f'Iteration {iter_id} finished. Process {error_process} failed.' )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='go2-gait')
    parser.add_argument('-e', '--exp_name', type=str, default=None)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=15000)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('-o', '--offline', action='store_true', default=False)
    parser.add_argument('--time', action='store_true', default=False)
    parser.add_argument('-p', '--ppo', action='store_false', default=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--disable', action='store_true', default=False)

    # Eval
    parser.add_argument('--num_eval_step', type=int, default=1000)
    parser.add_argument('-r', '--record', action='store_true', default=False)
    parser.add_argument('--record_length', help='unit: second', type=int, default=10)
    parser.add_argument('--resample_time', help='unit: second', type=float, default=2)

    args = parser.parse_args()

    if args.exp_name == None:
        args.exp_name = args.task
        now = datetime.datetime.now()
        args.exp_name += f'_{now.month}-{now.day}-{now.hour}-{now.minute}'

    args.cfg = args.task + '.yaml'
    args.robot = args.task.split('-')[0]
    
    if args.debug:
        args.vis = True
        args.offline = True
        args.num_envs = 1
        args.cpu = True

    if not torch.cuda.is_available():
        args.cpu = True

    main(args)
