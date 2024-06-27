from . import optimizer as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import math 

def lr_factory(model, cfg, steps_in_epoch, lr_policy):
    optimizer = optim.construct_optimizer(model, cfg)
    total_steps = cfg.SOLVER.MAX_EPOCH * steps_in_epoch

    if lr_policy == "cosine":
        slow_fast_scheduler = CosineAnnealingLR(
            optimizer, cfg.SOLVER.MAX_EPOCH * steps_in_epoch, last_epoch=-1
        )
    elif lr_policy == "constant":
        slow_fast_scheduler = LambdaLR(optimizer, lr_lambda=lambda x: 1)
    elif lr_policy == "cosine_warmup":
        slow_fast_scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=cfg.SOLVER.WARMUP_STEPS * steps_in_epoch,
            t_total=total_steps,
        )
    elif lr_policy == "linear_warmup":
        slow_fast_scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=cfg.SOLVER.WARMUP_STEPS,
            t_total=total_steps,
        )
    else:

        def lr_lambda(step):
            return optim.get_epoch_lr(step / steps_in_epoch, cfg)

        slow_fast_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    scheduler = {"scheduler": slow_fast_scheduler, "interval": "step"}
    return [optimizer], [scheduler]


class WarmupLinearSchedule(LambdaLR):
    """Linear warmup and then linear decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0,
            float(self.t_total - step)
            / float(max(1.0, self.t_total - self.warmup_steps)),
        )


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )
