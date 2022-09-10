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
        cfg_path = args.cfg
        try:
            shutil.copy(args.cfg, os.path.join(args.logdir, 'cfg_copy.yml'))
        except shutil.SameFileError:
            pass
    else:
        cfg_path = os.path.join(args.logdir, 'cfg_copy.yml')

    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = dict2namespace(cfg_dict)

    log_file = os.path.join(args.logdir, f'{args.mode}_log.txt')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=log_file, mode='w')
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel('INFO')

    agent = AGENT(cfg)
    if args.mode == 'train':
        agent.train(cfg, args.logdir, args.resume)
    elif args.mode == 'eval':
        agent.eval(cfg, args.logdir, ckpt_num=args.eval_ckpt)
    else:
        raise ValueError(f'Mode {args.mode} not recognized')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/config.yml')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--eval-ckpt', type=int, default=None)
    args = parser.parse_args()
    main(args)
