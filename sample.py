import argparse
import datetime

from scripts.manipulation.franka_sample import main as franka

MAIN_FUNCS = {
    'franka': franka,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='go2-jump')
    parser.add_argument('-e', '--exp_name', type=str, default=None)
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=10000)
    parser.add_argument('--num_eval_envs', type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument("--time_emb", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_emb", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])

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

    args.num_eval_envs = min(args.num_eval_envs, args.num_envs)

    main = MAIN_FUNCS[robot]
    main(args)


'''
# training
python train_backflip.py -e EXP_NAME

# evaluation
python eval_backflip.py -e EXP_NAME --ckpt NUM_CKPT
'''