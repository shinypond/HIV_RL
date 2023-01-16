from typing import *
import os
import argparse
import logging
import yaml
from torch.utils.tensorboard import SummaryWriter
from envs.hiv_v1 import make_HIV_env
from models.dqn.agent import DQNAgent
from configs import cfg


def main(args):

    # Define agent
    env = make_HIV_env(is_test=False, **cfg['env'])
    test_env = make_HIV_env(is_test=True, **cfg['env'])
    writer = SummaryWriter(args.tb_dir)
    if args.model == 'dqn':
        _agent = DQNAgent

    agent = _agent(
        env=env,
        test_env=test_env,
        writer=writer,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        load_ckpt=(args.resume) or (args.mode == 'test'),
        **cfg[f'{args.model}_agent'],
    )

    # Training Configuraiton
    cfg['train'] = dict(
        max_episodes = args.max_episodes,
        log_freq = 1,
        test_freq = 1,
        save_freq = 50,
        img_dir = args.img_dir,
    )

    # Save configs
    with open(os.path.join(args.log_dir, 'configs.yml'), 'w') as f:
        yaml.dump(cfg, f)

    # Train agent
    if args.mode == 'train':
        agent.train(**cfg['train'])

    # Test agent
    elif args.mode == 'test':
        agent.test(agent.init_episode - 1, args.img_dir)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model', type=str, default='dqn', choices=['dqn'])
    parser.add_argument('--max-episodes', type=int, default=2000)
    parser.add_argument('--resume', action='store_true', default=False)

    args = parser.parse_args()

    # Set GPU num
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'

    # Logging
    args.log_dir = os.path.join('logs', args.exp)
    os.makedirs(args.log_dir, exist_ok=True)
    args.tb_dir = os.path.join(args.log_dir, 'tb')
    os.makedirs(args.tb_dir, exist_ok=True)
    args.ckpt_dir = os.path.join(args.log_dir, 'ckpt')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.img_dir = os.path.join(args.log_dir, 'img')
    os.makedirs(args.img_dir, exist_ok=True)
    if args.mode == 'train':
        logging.basicConfig(
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{os.path.join(args.log_dir, "train_log.txt")}', mode='w'),
            ],
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.INFO,
        )
    main(args)