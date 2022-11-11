import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type

import torch

from pytorch_lightning import LightningDataModule
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn

import numpy as np
from pase_eeg.lit_modules.simple_classifier_lit import EEGBCIIV2aDataLit
from sklearn.model_selection import LeaveOneOut

import time
import wandb
import logging
from .utils import random_string

loger = logging.getLogger("lightning")
loger.setLevel("INFO")


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


class EEGBCIIV2aDataLit_CV(BaseKFoldDataModule):
    def __init__(
        self,
        data_path: str,
        channels_config: str,
        patients_list: List[int],
        batch_size: int = 32,
        transforms=[],
    ):
        super().__init__()

        self.data_path = data_path
        self.channels_config = channels_config
        self.patients_list = np.array(patients_list)
        self.batch_size = batch_size
        self.transforms = transforms

    def setup(self, stage: Optional[str] = None):
        self.setup_folds()
        self.setup_fold_index(0)

        print(self)

    def setup_folds(self) -> None:
        splitter = LeaveOneOut()
        self.num_folds = splitter.get_n_splits(self.patients_list)

        self.splits = [split for split in splitter.split(self.patients_list)]
        return self.num_folds

    def setup_fold_index(self, fold_index: int) -> None:
        self.dataset = EEGBCIIV2aDataLit(
            self.data_path,
            self.channels_config,
            self.patients_list[self.splits[fold_index][0]],
            self.patients_list[self.splits[fold_index][1]],
            self.batch_size,
            self.transforms,
        )
        self.dataset.setup()

    def train_dataloader(self):
        return self.dataset.train_dataloader()

    def val_dataloader(self):
        return self.dataset.val_dataloader()


class KFoldLoop(Loop):
    def __init__(self) -> None:
        super().__init__()
        self.current_fold: int = 0
        self.num_folds = sys.maxsize

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.num_folds = self.trainer.datamodule.num_folds
        loger.info(f"NUM Folds : {self.num_folds}")
        self.lightning_module_state_dict = deepcopy(
            self.trainer.lightning_module.state_dict()
        )

        self.random_id = random_string()

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        loger.info(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

        wandb.init(project="pase_eeg", group=f"G-{self.random_id}", reinit=True)
        assert wandb.run is not None

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()
        self.fit_loop.run()
        self.current_fold += 1

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""  # self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]
