import torch


class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Note: Implementation is origially from :
            https://github.com/ildoonet/pytorch-gradual-warmup-lr

    Parameters
    ----------
    optimizer : torch.optim
        Wrapped optimizer.
    warmup_steps : int
        warmup duration. target learning rate is reached at warmup_steps, gradually.
    after_scheduler : _LRScheduler
        after warmup_steps, use this scheduler(eg. ReduceLROnPlateau)
    multiplier : float, optional
        target learning rate = base lr * multiplier if multiplier > 1.0.
        if multiplier = 1.0, lr starts from 0 and ends up with the base_lr, by default 1.0

    Raises
    ------
    ValueError
        multiplier should be greater than or equal to 1.
    """

    def __init__(self, optimizer, warmup_steps, after_scheduler, multiplier=1.0):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_steps:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                if hasattr(self.after_scheduler, "get_last_lr"):
                    return self.after_scheduler.get_last_lr()
                elif hasattr(self.after_scheduler, "_last_lr"):
                    return self.after_scheduler._last_lr
                else:
                    return [
                        param_group_i["lr"]
                        for param_group_i in self.optimizer.param_groups
                    ]
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.warmup_steps + 1.0)
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None, metrics=None, *args, **kwargs):
        if self.finished and self.after_scheduler:
            if epoch is None:
                if metrics is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(None, metrics)
            else:
                if metrics is None:
                    self.after_scheduler.step(epoch - self.warmup_steps)
                else:
                    self.after_scheduler.step(epoch - self.warmup_steps, metrics)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
