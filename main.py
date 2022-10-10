from typing import *
import os
import argparse
import logging
import yaml
from torch.utils.tensorboard import SummaryWriter
from envs.hiv_env import make_HIV_env
from models.rainbow.agent import RainbowDQNAgent
from models.per_dqn.agent import PERDQNAgent


def main(args):

    cfg = {}

    # Agent Configuration
    # cfg['agent'] = dict(
    #     # Training parameters
    #     memory_size = int(1e6),
    #     batch_size = 1024,
    #     lr = 1e-4,
    #     l2_reg = 0.,
    #     target_update = 600 * 10,
    #     gamma = 1.,
    #     # Network parameters
    #     hidden_dim = 512,
    #     dropout = 0.0,
    #     # PER parameters
    #     alpha = 0.2,
    #     beta = 0.6,
    #     prior_eps = 1e-6,
    #     # Categorical DQN parameters
    #     v_min = 0.,
    #     v_max = 100.,
    #     atom_size = 51,
    #     # N-step Learning parameters
    #     n_step = 1,
    # )
    cfg['agent'] = dict(
        memory_size = int(1e6),
        batch_size = 2048,
        lr = 1e-4,
        l2_reg = 0.,
        grad_clip = 10.,
        target_update = 600 * 10,
        max_epsilon = 1.0,
        min_epsilon = 0.05,
        epsilon_decay = 1 / 2000,
        hidden_dim = 512,
        alpha = 0.2,
        beta = 0.6,
        prior_eps = 1e-6,
    )

    # Define agent
    env = make_HIV_env()
    test_env = make_HIV_env()
    writer = SummaryWriter(args.tb_dir)
    if args.model == 'rainbow':
        _agent = RainbowDQNAgent
    elif args.model == 'perdqn':
        _agent = PERDQNAgent

    agent = _agent(
        env=env,
        test_env=test_env,
        writer=writer,
        ckpt_dir=args.ckpt_dir,
        load_ckpt=(args.resume) or (args.mode == 'test'),
        **cfg['agent'],
    )

    # Training Configuraiton
    cfg['train'] = dict(
        max_episodes = args.max_episodes,
        log_freq = 1,
        test_freq = 1,
        save_freq = 10,
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
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model', type=str, required=True, choices=['rainbow', 'perdqn'])
    parser.add_argument('--max-episodes', type=int, default=10000)
    parser.add_argument('--resume', action='store_true', default=False)

    args = parser.parse_args()

    # Logging
    args.log_dir = os.path.join('logs', args.exp)
    os.makedirs(args.log_dir, exist_ok=True)
    args.tb_dir = os.path.join(args.log_dir, 'tb')
    os.makedirs(args.tb_dir, exist_ok=True)
    args.ckpt_dir = os.path.join(args.log_dir, 'ckpt')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    args.img_dir = os.path.join(args.log_dir, 'img')
    os.makedirs(args.img_dir, exist_ok=True)
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