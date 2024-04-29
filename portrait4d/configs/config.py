import os
import argparse
import torch

from .yacs import CfgNode as CN

# pylint: disable=redefined-outer-name

_C = CN()

def get_cfg_defaults():
    return _C.clone()


def parse_cfg(cfg):
    cfg.logdir = os.path.join('out', cfg.experiment)


def determine_primary_secondary_gpus(cfg):
    print("------------------ GPU Configurations ------------------")
    cfg.n_gpus = torch.cuda.device_count()
    if cfg.n_gpus > 0:
        all_gpus = list(range(cfg.n_gpus))
        cfg.primary_gpus = [0]
        if cfg.n_gpus > 1:
            cfg.secondary_gpus = [g for g in all_gpus]# if g not in cfg.primary_gpus]
        else:
            cfg.secondary_gpus = cfg.primary_gpus
        print(f"Primary GPUs: {cfg.primary_gpus}")
        print(f"Secondary GPUs: {cfg.secondary_gpus}")
    else:
        print(f"CPU job")
    print("--------------------------------------------------------")


def make_cfg(args):
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/default.yaml')
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.file_path = args.cfg
    parse_cfg(cfg)

    # determine_primary_secondary_gpus(cfg)
        
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg", default='configs/genhead-ffhq512.yaml', type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

cfg = make_cfg(args)