import argparse
import datetime

from scripts.locomotion.go2_eval import main as go2

MAIN_FUNCS = {
    'go2': go2,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='go2-jump')
    parser.add_argument('-e', '--exp_name', type=str, default=None)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-r', '--record', action='store_true', default=False)
    parser.add_argument('-H', '--headless', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=1)
    parser.add_argument('--td', action='store_true', default=False)
    parser.add_argument('--ckpt', type=int, default=999)
    parser.add_argument('--record_length', help='unit: second', type=int, default=3)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--resample_time', help='unit: second', type=float, default=None)
    args = parser.parse_args()

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