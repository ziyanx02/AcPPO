import argparse
import datetime
import yaml
import re
import statistics

import logging
import os

import torch.multiprocessing as mp
import torch
from reward_tuning.prompt import INITIAL_USER, INITIAL_SYSTEM, JUDGE_SYSTEM, JUDGE_USER
from reward_tuning.prompt import POLICY_FEEDBACK, CODE_FEEDBACK, CODE_OUTPUT_TIP
from reward_tuning.example import RESPONSE_SAMPLE_REWARD
from reward_tuning.client import Client

from scripts.train import train
from scripts.eval import eval

def setup_logger(exp_name):
    log_dir = 'logs_run'
    logging.basicConfig(
        filename=f'{log_dir}/{exp_name}.log',
        level=logging.DEBUG,            
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',  
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger()

def train_eval(return_queue, args, response, iter_id, sample_id, train_cfg, env_cfg, tune_cfg, device_id):
    logger = setup_logger(args.exp_name)
    logger.info(f"Sample {sample_id} training start on device {device_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    train_queue = mp.Queue()
    eval_queue = mp.Queue()

    train_process = mp.Process(
        target=train,
        args=(train_queue, args, response, iter_id, sample_id, train_cfg, env_cfg, tune_cfg),
    )
    train_process.start()
    train_process.join()
    try:
        train_return = train_queue.get_nowait()
    except:
        logger.error(f"Training process for sample {sample_id} did not return any data.")
        raise RuntimeError(f"Training process for sample {sample_id} did not return any data.")

    eval_process = mp.Process(
        target=eval,
        args=(eval_queue, args, train_return['exp_name'])
    )
    eval_process.start()
    eval_process.join()

    try:
        eval_return = eval_queue.get_nowait()
    except:
        logger.error(f"Evaluation process for sample {sample_id} did not return any data.")
        raise RuntimeError(f"Evaluation process for sample {sample_id} did not return any data.")

    return_queue.put({
        'train': train_return,
        'eval': eval_return,
    })

    logger.info(f"Sample {sample_id} finish")

def get_best(client, results):
    # LLM-based judgement from evaluation result
    eval_result = ''
    for result in results:
        idx = result['train']['sample_id']
        metric = result['eval']['metric']
        eval_result += f'Index: {idx}\n'
        for key in metric.keys():
            eval_result += f'   {key}: {metric[key]:.3f}\n'

    message = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER + eval_result}
    ]

    response = client.response(message)
    match = re.search(r'```best\n(.*?)\n```', response, re.DOTALL)
    if match : 
        idx = int(match.group(1))
    else:
        raise ValueError("No best index found in response")

    for result in results:
        if idx == result['train']['sample_id']:
            return result
    raise ValueError("Best index not match")

def get_reward_reflection(result):
    content = "" 
    content += POLICY_FEEDBACK.format(epoch_freq=result['train']['log_frequency'])
    log_dict = result['train']['train_log']
    for key in log_dict[list(log_dict.keys())[0]]:
        values = [log_dict[i][key] for i in log_dict.keys()]
        content += f"{key}: {values}, Max {max(values)}, Mean {statistics.mean(values)}, Min {min(values)}"
    content += CODE_FEEDBACK.format(max_episode_length=result['train']['max_episode_length'])
    content += CODE_OUTPUT_TIP

    return content

def main(args):
    mp.set_start_method("spawn", force=True)

    logger = setup_logger(args.exp_name)

    client_reward = Client(disable=args.disable, template=RESPONSE_SAMPLE_REWARD)
    client_judge = Client()

    base_message = [
        {"role": "system", "content": INITIAL_SYSTEM},
        {"role": "user", "content": INITIAL_USER}
    ]

    with open(f'./cfgs/{args.cfg}', 'r') as file:
        cfg = yaml.safe_load(file)
    train_cfg = cfg['learning']
    env_cfg = cfg['environment']
    tune_cfg = cfg['reward_tuning']

    best_result_list = []

    for iter_id in range(tune_cfg['num_iterations']):
        logger.info(f"Iteration {iter_id} start")
        return_queue = mp.Queue()
        process = []

        for sample_id in range(tune_cfg['num_samples']):
            # try:
            response = client_reward.response(base_message)
            sample_process = mp.Process(
                target=train_eval,
                args=(return_queue, args, response, iter_id, sample_id, train_cfg, env_cfg, tune_cfg, (sample_id + 1) % torch.cuda.device_count())
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
        
        logger.info(f"Iteration {iter_id} finished. Process {error_process} failed.")

        results = []
        while not return_queue.empty():
            results.append(return_queue.get())
        best_result = get_best(client_judge, results)
        best_result_list.append(best_result)

        logger.info(f"Evaluation finished. Sample {best_result['train']['sample_id']} wins. ")
        logger.debug(f"Details of best reward:\n{best_result}")

        assist_message = {"role": "assistant", "content": best_result['train']['response']['raw']}
        user_message = {"role": "user", "content": get_reward_reflection(best_result)}

        if len(base_message) == 2:
            base_message += [assist_message, user_message]
        else :
            base_message[-2:] = [assist_message, user_message]
    
    logger.info(f"Finish all iterations.")
    best_of_all = get_best(client_judge, best_result_list)
    logger.info(f"Best result of all reward parameters: {best_of_all['train']['exp_name']}")


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

    print(f'Exp_name : {args.exp_name}')

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
