
import os
import os.path as osp

from omegaconf import OmegaConf


def load_config(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    if '_base_' in cfg:
        if isinstance(cfg._base_, str):
            base_cfg = OmegaConf.load(osp.join(osp.dirname(cfg_file), cfg._base_))
        else:
            base_cfg = OmegaConf.merge(OmegaConf.load(f) for f in cfg._base_)
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg

def get_config(args):
    cfg = load_config(args.config)
    OmegaConf.set_struct(cfg, True)

    if hasattr(args, 'output_dir') and args.output_dir:
        cfg.output_dir = args.output_dir
   
    if hasattr(args, 'wandb') and args.wandb:
        cfg.wandb = args.wandb

    if hasattr(args, 'testonly') and args.testonly:
        cfg.test.testonly = args.testonly

    if hasattr(args, 'local_rank'):
        cfg.local_rank = args.local_rank

    OmegaConf.set_readonly(cfg, True)

    return cfg
