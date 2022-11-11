from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
import torch
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.loggers import TensorBoardLogger

from tqdm import tqdm
from .multi_optim import MultipleScheduler, MultipleOptimizer
from pase_eeg.manual_trainer.loggers.tensorboard_logger import TensorBoardManualLogger


class Trainer:
    def __init__(
        self,
        lit_model,
        lit_dataloader,
        max_epochs: int = 10,
        callbacks: list = None,
        device: str = "cpu",
        logger: bool = False,
    ):

        self.lit_model = lit_model
        self.lit_dataloader = lit_dataloader
        self.max_epochs = max_epochs
        self.callbacks = callbacks
        self.device = device

        # init Logger
        self.logger = TensorBoardManualLogger()
        self.lit_model.log = self.logger.log

        self.setup_on_init()

    def setup_on_init(self):
        self.get_optim_config()
        self.lit_dataloader.setup()

        # Transfer our model to desirable devices
        self.lit_model.model.to(device=self.device)

    def get_optim_config(self):
        conf = self.lit_model.configure_optimizers()
        if isinstance(conf, tuple) and len(conf) == 2:
            optimizers, lr_schedulers = conf
        elif isinstance(conf, list):
            optimizers = conf

        if isinstance(optimizers, list):
            self.optimizer = MultipleOptimizer(optimizers)
        elif isinstance(optimizers, torch.optim.optimizer.Optimizer):
            self.optimizer = optimizers

        if isinstance(lr_schedulers, list):
            self.lr_scherduler = MultipleScheduler(lr_schedulers)
        elif lr_schedulers is not None:
            self.lr_scherduler = lr_schedulers

    def on_epoch_end(self):
        if self.callbacks:
            for clbk in self.callbacks:
                if hasattr(clbk, "on_epoch_end"):
                    clbk.on_epoch_end(self)

    def fit(self):
        for epoch in range(self.max_epochs):  # loop over the dataset multiple times
            print("\nEpoch ", epoch)
            self._train(self.lit_dataloader.train_dataloader(), epoch)
            self._evaluate(self.lit_dataloader.val_dataloader(), epoch)
            self.on_epoch_end()

    def _train(self, dl, epoch: Optional[int] = None):
        print("Training...")
        for idx, batch in tqdm(enumerate(dl), total=len(dl), ncols=100):
            # Transfer Our Barch to device
            batch = move_data_to_device(batch, self.device)
            self.optimizer.zero_grad()
            loss = self.lit_model.training_step(
                batch=batch, batch_idx=idx, num_epoch=epoch
            )
            loss.backward()
            self.optimizer.step()

        self.lr_scherduler.step()

    def _evaluate(self, dl, epoch: Optional[int] = None):
        print("Validating...")
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dl), total=len(dl), ncols=100):
                # Transfer Our Barch to device
                batch = move_data_to_device(batch, self.device)
                self.lit_model.validation_step(
                    batch=batch, batch_idx=idx, num_epoch=epoch
                )
