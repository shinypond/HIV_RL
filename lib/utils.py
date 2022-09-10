import os
import argparse
import torch


def load_ckpt(ckpt_dir, info, device, train=True, ckpt_num=None):
    if not os.path.exists(ckpt_dir):
        return info
    fname = 'ckpt.pt' if ckpt_num is None else f'ckpt_{ckpt_num}.pt'
    loaded_info = torch.load(
        os.path.join(ckpt_dir, fname),
        map_location=device
    )
    info['policy_net'].load_state_dict(loaded_info['policy_net'])
    info['policy_net'].train()
    info['episode'] = loaded_info['episode']
    if train:
        info['target_net'].load_state_dict(loaded_info['target_net'])
        info['target_net'].eval()
        info['optimizer'].load_state_dict(loaded_info['optimizer'])
        info['scheduler'].load_state_dict(loaded_info['scheduler'])
        info['memory'] = loaded_info['memory']
    return info

        
def save_ckpt(ckpt_dir, info, archive=False):
    saved_info = {
        'policy_net': info['policy_net'].state_dict(),
        'target_net': info['target_net'].state_dict(),
        'optimizer': info['optimizer'].state_dict(),
        'scheduler': info['scheduler'].state_dict(),
        'memory': info['memory'],
        'episode': info['episode'],
    }
    if archive:
        ckpt = os.path.join(ckpt_dir, f'ckpt_{info["episode"]}.pt')
    else:
        ckpt = os.path.join(ckpt_dir, 'ckpt.pt')
    torch.save(saved_info, ckpt)


def dict2namespace(cfg):
    namespace = argparse.Namespace()
    for key, value in cfg.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def str2bool(x):
    if isinstance(x, bool):
        return x
    else:
        assert isinstance(x, str)
        if x.lower() in ['yes', 'true', 't', 'y', '1']:
            return True
        elif x.lower() in ['no', 'false', 'f', 'n', '0']:
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

