import argparse
import datetime

from scripts.locomotion.g1_train import main as g1
from scripts.locomotion.go2_train import main as go2
from scripts.manipulation.franka_train import main as franka

MAIN_FUNCS = {
    'g1': g1,
    'go2': go2,
    'franka': franka,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='go2-gait')
    parser.add_argument('-e', '--exp_name', type=str, default=None)
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=15000)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('-o', '--offline', action='store_true', default=False)
    parser.add_argument('--time', action='store_true', default=False)
    parser.add_argument('-p', '--ppo', action='store_true', default=False)

    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--ckpt', type=int, default=1000)
    args = parser.parse_args()

    if args.exp_name == None:
        args.exp_name = args.task
        now = datetime.datetime.now()
        args.exp_name += f'_{now.month}-{now.day}-{now.hour}-{now.minute}'

    robot, task = args.task.split('-')
    args.task = task

    main = MAIN_FUNCS[robot]
    main(args)


'''
# training
python train_backflip.py -e EXP_NAME

# evaluation
python eval_backflip.py -e EXP_NAME --ckpt NUM_CKPT
'''