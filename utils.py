import os
import torch
import logging
import functools
import sys
from typing import List, Union
from torch.optim import Optimizer
from cosine_lr import CosineLRScheduler

def create_scheduler(
        optimizer: Optimizer,
        sched: str = 'cosine',
        num_epochs: int = 300,
        decay_epochs: int = 90,
        decay_milestones: List[int] = (90, 180, 270),
        cooldown_epochs: int = 0,
        min_lr: float = 0,
        warmup_lr: float = 1e-5,
        warmup_epochs: int = 0,
        warmup_prefix: bool = False,
        noise: Union[float, List[float]] = None,
        noise_pct: float = 0.67,
        noise_std: float = 1.,
        noise_seed: int = 42,
        cycle_mul: float = 1.,
        cycle_decay: float = 0.1,
        cycle_limit: int = 1,
        k_decay: float = 1.0,
        step_on_epochs: bool = True,
        updates_per_epoch: int = 0,
):
    t_initial = num_epochs
    warmup_t = warmup_epochs
    decay_t = decay_epochs
    cooldown_t = cooldown_epochs

    if not step_on_epochs:
        assert updates_per_epoch > 0, 'updates_per_epoch must be set to number of dataloader batches'
        t_initial = t_initial * updates_per_epoch
        warmup_t = warmup_t * updates_per_epoch
        decay_t = decay_t * updates_per_epoch
        decay_milestones = [d * updates_per_epoch for d in decay_milestones]
        cooldown_t = cooldown_t * updates_per_epoch

    # warmup args
    warmup_args = dict(
        warmup_lr_init=warmup_lr,
        warmup_t=warmup_t,
        warmup_prefix=warmup_prefix,
    )

    # setup noise args for supporting schedulers
    if noise is not None:
        if isinstance(noise, (list, tuple)):
            noise_range = [n * t_initial for n in noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = noise * t_initial
    else:
        noise_range = None
    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=noise_pct,
        noise_std=noise_std,
        noise_seed=noise_seed,
    )

    # setup cycle args for supporting schedulers
    cycle_args = dict(
        cycle_mul=cycle_mul,
        cycle_decay=cycle_decay,
        cycle_limit=cycle_limit,
    )

    lr_scheduler = None
    if sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=min_lr,
            t_in_epochs=step_on_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
        )

    if hasattr(lr_scheduler, 'get_cycle_length'):
        # for cycle based schedulers (cosine, tanh, poly) recalculate total epochs w/ cycles & cooldown
        t_with_cycles_and_cooldown = lr_scheduler.get_cycle_length() + cooldown_t
        if step_on_epochs:
            num_epochs = t_with_cycles_and_cooldown
        else:
            num_epochs = t_with_cycles_and_cooldown // updates_per_epoch

    return lr_scheduler, num_epochs

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name='', default_level=logging.INFO):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(default_level)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(default_level)
        console_handler.setFormatter(
            logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)
    
    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(default_level)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger
