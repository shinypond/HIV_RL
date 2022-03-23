import os
import argparse
import yaml
import logging
import shutil

from lib.agent import AGENT
from lib.utils import dict2namespace, str2bool


def main(args):
    os.makedirs(args.logdir, exist_ok=True)
    
    if not args.resume:
        config_path = args.config
        try:
            shutil.copy(args.config, os.path.join(args.logdir, 'config_copy.yml'))
        except shutil.SameFileError:
            pass
    else:
        config_path = os.path.join(args.logdir, 'config_copy.yml')

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = dict2namespace(config_dict)

    log_file = os.path.join(args.logdir, f'{args.mode}_log.txt')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=log_file, mode='w')
    formatter = logging.Formatter(
        '%(levelname)s - %(filename)s - %(asctime)s - %(message)s'
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel('INFO')

    agent = AGENT(config)
    if args.mode == 'train':
        agent.train(config, args.logdir, args.resume)
    elif args.mode == 'eval':
        agent.eval(config, args.logdir, ckpt_num=config.eval.ckpt_num)
    else:
        raise ValueError(f'Mode {args.mode} not recognized')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./lib/configs/config.yml')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--resume', type=str2bool, default=False)
    args = parser.parse_args()
    main(args)
