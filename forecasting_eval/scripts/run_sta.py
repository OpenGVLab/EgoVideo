import os
import pickle
import pprint
import sys
sys.path.append("/mnt/cache/xingsen/ego4d")
import ego4d.utils.logging as logging
import numpy as np
import torch
from ego4d.tasks.short_term_anticipation import ShortTermAnticipationTask
from ego4d.utils.c2_model_loading import get_name_convert_func
from ego4d.utils.parser import load_config, parse_args
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from scripts.slurm import copy_and_run_with_config
from pytorch_lightning.plugins import DDPPlugin
from collections import OrderedDict



logger = logging.get_logger(__name__)


def main(cfg):
    seed_everything(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Run with config:")
    logger.info(pprint.pformat(cfg))

    # Choose task type based on config.
    if cfg.DATA.TASK == "short_term_anticipation":
        TaskType = ShortTermAnticipationTask

    task = TaskType(cfg)

    # Load model from checkpoint if checkpoint file path is given.
    ckp_path = cfg.CHECKPOINT_FILE_PATH
    if len(ckp_path) > 0:
        if cfg.CHECKPOINT_VERSION == "caffe2":
            with open(ckp_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            state_dict = data["blobs"]
            fun = get_name_convert_func()
            state_dict = {
                fun(k): torch.from_numpy(np.array(v))
                for k, v in state_dict.items()
                if "momentum" not in k and "lr" not in k and "model_iter" not in k
            }

            if not cfg.CHECKPOINT_LOAD_MODEL_HEAD:
                state_dict = {k: v for k, v in state_dict.items() if "head" not in k}
            print(task.model.load_state_dict(state_dict, strict=False))
            print(f"Checkpoint {ckp_path} loaded")
        elif cfg.CHECKPOINT_VERSION == "pytorch":
            checkpoint = torch.load(ckp_path, map_location="cpu")
            print("Load ckpt from {}".format(ckp_path))
            checkpoint_model = None
            model_keys = ['model','module']
            for model_key in model_keys:
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break
            if checkpoint_model is None:
                checkpoint_model = checkpoint
            state_dict = task.model.state_dict()
            try:
                for k in ['head.weight', 'head.bias']:  #
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
            except KeyError:
                del checkpoint_model['head.weight']
                del checkpoint_model['head.bias']
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif key.startswith('encoder.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    new_dict[key] = checkpoint_model[key]
            checkpoint_model = new_dict
            if 'pos_embed' in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
                num_patches = task.model.patch_embed.num_patches  #
                num_extra_tokens = task.model.pos_embed.shape[-2] - num_patches  # 0/1

                # height (== width) for the checkpoint position embedding 
                orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (
                            cfg.DATA.NUM_FRAMES // task.model.patch_embed.tubelet_size)) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int((num_patches // (cfg.DATA.NUM_FRAMES // task.model.patch_embed.tubelet_size)) ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    # B, L, C -> BT, H, W, C -> BT, C, H, W
                    pos_tokens = pos_tokens.reshape(-1, cfg.DATA.NUM_FRAMES // task.model.patch_embed.tubelet_size, orig_size,
                                                    orig_size, embedding_size)
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1,
                                                                        cfg.DATA.NUM_FRAMES // task.model.patch_embed.tubelet_size,
                                                                        new_size, new_size, embedding_size)
                    pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    checkpoint_model['pos_embed'] = new_pos_embed
            my_load_state_dict(task.model, checkpoint_model)
        elif cfg.CHECKPOINT_VERSION == "uniformer":
            print("ckpt has beed loaded!")
        elif cfg.DATA.CHECKPOINT_MODULE_FILE_PATH != "":

            # Load slowfast weights into backbone submodule
            ckpt = torch.load(
                cfg.DATA.CHECKPOINT_MODULE_FILE_PATH,
                map_location=lambda storage, loc: storage,
            )

            def remove_first_module(key):
                return ".".join(key.split(".")[1:])

            state_dict = {
                remove_first_module(k): v
                for k, v in ckpt["state_dict"].items()
                if "head" not in k
            }
            missing_keys, unexpected_keys = task.model.backbone.load_state_dict(
                state_dict, strict=False
            )

            # Ensure only head key is missing.
            assert len(unexpected_keys) == 0
            assert all(["head" in x for x in missing_keys])
        else:
            # Load all child modules except for "head" if CHECKPOINT_LOAD_MODEL_HEAD is
            # False.
            pretrained = TaskType.load_from_checkpoint(ckp_path)
            state_dict_for_child_module = {
                child_name: child_state_dict.state_dict()
                for child_name, child_state_dict in pretrained.model.named_children()
            }
            for child_name, child_module in task.model.named_children():
                if not cfg.CHECKPOINT_LOAD_MODEL_HEAD and "head" in child_name:
                    continue

                state_dict = state_dict_for_child_module[child_name]
                child_module.load_state_dict(state_dict)

    checkpoint_callback = ModelCheckpoint(
        monitor=task.checkpoint_metric, mode="min", save_last=True, save_top_k=1
    )
    if cfg.ENABLE_LOGGING:
        args = {"callbacks": [LearningRateMonitor(), checkpoint_callback]}
    else:
        args = {"logger": False, "callbacks": checkpoint_callback}
    print(cfg.ENABLE_LOGGING)
    trainer = Trainer(
        gpus=cfg.NUM_GPUS,
        num_nodes=cfg.NUM_SHARDS,
        accelerator=cfg.SOLVER.ACCELERATOR,
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        num_sanity_val_steps=3,
        benchmark=True,
        log_gpu_memory="min_max",
        replace_sampler_ddp=False,
        fast_dev_run=cfg.FAST_DEV_RUN,
        default_root_dir=cfg.OUTPUT_DIR,
        plugins=DDPPlugin(find_unused_parameters=True),
        **args,
    )

    if cfg.TRAIN.ENABLE and cfg.TEST.ENABLE:
        trainer.fit(task)

        # Calling test without the lightning module arg automatically selects the best
        # model during training.
        return trainer.test()

    elif cfg.TRAIN.ENABLE:
        return trainer.fit(task)

    elif cfg.TEST.ENABLE:
        return trainer.test(task)

def my_load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))



if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    if args.on_cluster:
        copy_and_run_with_config(
            main,
            cfg,
            args.working_directory,
            job_name=args.job_name,
            time="72:00:00",
            partition="learnfair",
            gpus_per_node=cfg.NUM_GPUS,
            ntasks_per_node=cfg.NUM_GPUS,
            cpus_per_task=10,
            mem="470GB",
            nodes=cfg.NUM_SHARDS,
            constraint="volta32gb",
        )
    else:  # local
        main(cfg)
