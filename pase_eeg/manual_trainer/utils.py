from pase_eeg.utils.lr_scheduler import GradualWarmupScheduler
from .multi_optim import MultipleScheduler, MultipleOptimizer

import torch


def configure_optimizers(self, model, learning_rate: float = 3e-4):
    params = list(model.parameters())
    optimizers = MultipleOptimizer(
        torch.optim.SGD(
            [params[0], params[1]],
            lr=learning_rate * 50,
            momentum=0.9,
            nesterov=True,
        ),
        torch.optim.SGD(params[2:], lr=learning_rate, momentum=0.9, nesterov=True),
    )

    schedulers = MultipleScheduler(
        GradualWarmupScheduler(
            self.optimizers[0],
            80,
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers[0], T_max=220, eta_min=1e-9
            ),
        ),
        GradualWarmupScheduler(
            self.optimizers[1],
            80,
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers[1], T_max=220, eta_min=1e-9
            ),
        ),
    )
