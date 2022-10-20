from typing import Any, Dict, List, Optional, Tuple, Union
from torch import Tensor

import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.trainer.states import TrainerFn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .utils import (
    eeg_electrode_configs,
    read_json_config,
    transforms_from_worker_configs,
    instantiate_class,
)

from ..nn.models.embedder import EEGEmb
from ..nn.models.eegnet import EEGNetv2Emb
from ..nn.modules.minions import minion_maker
from ..data.synthetic_dataset import EEGSyntheticDataset
from ..data.chb_mit import CHBMITDataset
from ..data.bci import BCI2aDataset
from ..data.transforms import Compose

from ..utils.lr_scheduler import GradualWarmupScheduler
from ..data.LEE import LEEDataset


class PASE(LightningModule):
    def __init__(
        self,
        channels_config: str,
        emb_dim: int,
        learning_rate: float = 3e-4,
        min_learning_rate: float = 1e-7,
        workers_config: str = None,
        pretrained_backend_weights_path: str = None,
        logger: dict = {"proj": "my_project", "wb_group": "pase_eeg", "exp": "0"},
    ) -> None:

        super().__init__()
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.emb_dim = emb_dim
        self.workers_config_path = workers_config

        (
            self.eeg_electrode_positions,
            self.eeg_electrods_plane_shape,
        ) = eeg_electrode_configs(channels_config)

        # self.model = EEGEmb(
        #     channel_positions=self.eeg_electrode_positions,
        #     channels_plane_shape=self.eeg_electrods_plane_shape,
        #     emb_dim=emb_dim,
        # )

        self.model = EEGNetv2Emb(
            emb_dim=emb_dim,
        )

        if pretrained_backend_weights_path is not None:
            self.model = self._load_weigths(self.model, pretrained_backend_weights_path)

    def _load_weigths(self, model, path):
        checkpoint = torch.load(path)
        state_dict = {
            key.replace("model.", ""): value
            for key, value in checkpoint["state_dict"].items()
            if "model." in key
        }
        model.load_state_dict(state_dict)

        return model

    def setup(self, stage: Optional[str] = None):
        if stage in [TrainerFn.FITTING, TrainerFn.VALIDATING, TrainerFn.TUNING]:
            self.setup_workers()

    def setup_workers(self):
        worker_configs = read_json_config(self.workers_config_path)
        self.workers = nn.ModuleDict()
        for conf in worker_configs:
            conf["in_shape"] = self.emb_dim
            conf["channels"] = self.eeg_electrode_positions
            conf.pop("transform", None)
            minion = minion_maker(conf)
            self.workers[conf["name"]] = minion

    def forward(self, x: Dict[str, Tensor]) -> Tensor:
        embeddings = self.model(x, device=self.device)
        return embeddings

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Union[Any, Dict[str, Tensor]]]],
        name: str = "train",
    ):
        x, y = batch
        x = torch.permute(
            torch.vstack(list(map(lambda a: a.unsqueeze(0), x.values()))),
            (1, 2, 3, 0),
        )
        embeddings = self(x)

        # regression training step
        losses = {}
        total_loss = 0
        for key in self.workers.keys():
            logits = self.workers[key](embeddings)
            losses[key] = self.workers[key].loss_weight * self.workers[key].loss(
                logits, y[key]
            )
            total_loss = total_loss + losses[key]

            self.log(f"{name}_{key}_loss", losses[key], prog_bar=True)
        self.log(f"total_loss", total_loss, prog_bar=True)

        return total_loss

    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Union[Any, Dict[str, Tensor]]]],
        batch_idx: int,
        optimizer_idx: int,
    ):
        return self._step(batch, name="train")

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Union[Any, Dict[str, Tensor]]]],
        batch_idx: int,
    ):
        return self._step(batch, name="val")

    def configure_optimizers(self):
        params = list(self.parameters())
        optimizers = []
        schedulers = []

        optimizers.append(
            torch.optim.SGD([params[0], params[1]], lr=0.1, momentum=0.9, nesterov=True)
        )
        optimizers.append(
            torch.optim.SGD(
                params[2:], lr=self.learning_rate, momentum=0.9, nesterov=True
            )
        )

        schedulers.append(
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[0], T_max=290, eta_min=1e-9
                ),
                # GradualWarmupScheduler(
                #     optimizer,
                #     80,
                #     torch.optim.lr_scheduler.CosineAnnealingLR(
                #         optimizer, T_max=220, eta_min=1e-9
                #     ),
                #     # torch.optim.lr_scheduler.ReduceLROnPlateau(
                #     #     optimizer, min_lr=self.min_learning_rate
                #     # ),
                # ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "total_loss",
                "strict": True,
            }
        )
        schedulers.append(
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[1], T_max=300, eta_min=1e-9
                ),
                # GradualWarmupScheduler(
                #     optimizer,
                #     80,
                #     torch.optim.lr_scheduler.CosineAnnealingLR(
                #         optimizer, T_max=220, eta_min=1e-9
                #     ),
                #     # torch.optim.lr_scheduler.ReduceLROnPlateau(
                #     #     optimizer, min_lr=self.min_learning_rate
                #     # ),
                # ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train_loss",
                "strict": True,
            }
        )
        return optimizers, schedulers

    def training_epoch_end(self, training_step_outputs):
        print("\n")

    def training_epoch_start(self, training_step_outputs):
        print("\n")


class PaseEEGSynthetichDataLit(LightningDataModule):
    def __init__(
        self,
        channels_config: str,
        batch_size: int = 32,
        workers_config: str = None,
    ):
        super().__init__()
        self.batch_size = batch_size

        self.eeg_electrode_positions, _ = eeg_electrode_configs(channels_config)

        worker_configs = read_json_config(workers_config)
        worker_transforms = transforms_from_worker_configs(worker_configs)

        transforms = list(map(instantiate_class, transforms))

        self.transforms = Compose(worker_transforms + transforms)
        print(self.transforms)

    def setup(self, stage: Optional[str] = None):
        self.dataset = EEGSyntheticDataset(
            eeg_electrode_positions=self.eeg_electrode_positions,
            transforms=self.transforms,  # ToTensor(device=torch.device("cuda:0")),
        )
        self.train_idx, self.test_idx, _, _ = train_test_split(
            list(range(len(self.dataset))),
            self.dataset.labels,
            stratify=self.dataset.labels,
            test_size=0.2,
        )
        print(self)

    def train_dataloader(self):
        return DataLoader(
            self.dataset.subset(self.train_idx),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.subset(self.test_idx),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        imgs = {
            key: torch.vstack([item[0][key].unsqueeze(0) for item in batch])
            for key in batch[0][0].keys()
        }
        trgts = {
            key: torch.vstack([item[1][key].squeeze() for item in batch])
            for key in batch[0][1].keys()
        }

        return [imgs, trgts]

    def __repr__(self) -> str:
        return (
            "#############################################################\n"
            + f"Dataset              : {self.dataset.__class__.__name__}\n"
            + f"# Total Samples      : {len(self.dataset)}\n"
            + f"# Train Samples      : {len(self.train_idx)}\n"
            + f"# Validation Samples : {len(self.test_idx)}\n"
            + "#############################################################\n"
        )


class PaseEEGCHBMITDataLit(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        channels_config: str,
        patient_list: List[int],
        length: int,
        batch_size: int = 32,
        workers_config: str = None,
        transforms=[],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.patient_list = patient_list
        self.length = length
        self.eeg_electrode_positions, _ = eeg_electrode_configs(channels_config)
        self.data_path = data_path

        worker_configs = read_json_config(workers_config)
        worker_transforms = transforms_from_worker_configs(worker_configs)

        transforms = list(map(instantiate_class, transforms))

        self.transforms = Compose(worker_transforms + transforms)
        print(self.transforms)

    def setup(self, stage: Optional[str] = None):
        self.dataset = CHBMITDataset(
            eeg_electrode_positions=self.eeg_electrode_positions,
            transforms=self.transforms,
            patient_list=self.patient_list,
            data_path=self.data_path,
            length=self.length,
            verbose=False,
        )

        self.train_idx, self.test_idx, _, _ = train_test_split(
            list(range(len(self.dataset))),
            self.dataset.labels,
            stratify=self.dataset.labels,
            test_size=0.2,
        )
        print(self)

    def train_dataloader(self):
        return DataLoader(
            self.dataset.subset(self.train_idx),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count(),
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.subset(self.test_idx),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count(),
            drop_last=True,
        )

    @staticmethod
    def collate_fn(batch):
        imgs = {
            key: torch.vstack([item[0][key].unsqueeze(0) for item in batch])
            for key in batch[0][0].keys()
        }
        trgts = {}
        sample_item_label = batch[0][1]
        for label_key in sample_item_label.keys():
            if isinstance(sample_item_label[label_key], dict):
                trgts[label_key] = {
                    key: torch.vstack(
                        [item[1][label_key][key].squeeze() for item in batch]
                    )
                    for key in sample_item_label[label_key].keys()
                }
            # else:
            # trgts[label_key] = torch.vstack([item[1][label_key] for item in batch]).squeeze()

        return [imgs, trgts]

    def __repr__(self) -> str:
        return (
            "#############################################################\n"
            + f"Dataset              : {self.dataset.__class__.__name__}\n"
            + f"# Total Samples      : {len(self.dataset)}\n"
            + f"# Train Samples      : {len(self.train_idx)}\n"
            + f"# Validation Samples : {len(self.test_idx)}\n"
            + f"# Negative Samples   : {self.dataset.get_class_distribution()[0]}\n"
            + f"# Positive Samples   : {self.dataset.get_class_distribution()[1]}\n"
            + "#############################################################\n"
        )


class PaseEEGLEEDataLit(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        channels_config: str,
        train_patients: List[int],
        test_patients: List[int],
        length: int,
        batch_size: int = 32,
        workers_config: str = None,
        transforms=[],
    ):
        super().__init__()
        self.data_path = data_path
        self.eeg_electrode_positions, _ = eeg_electrode_configs(channels_config)
        self.batch_size = batch_size
        self.train_patients = train_patients
        self.test_patients = test_patients
        self.length = length

        worker_configs = read_json_config(workers_config)
        worker_transforms = transforms_from_worker_configs(worker_configs)

        transforms = list(map(instantiate_class, transforms))

        self.transforms = Compose(worker_transforms + transforms)
        print(self.transforms)

    def setup(self, stage: Optional[str] = None):
        self.dataset = LEEDataset(
            eeg_electrode_positions=self.eeg_electrode_positions,
            data_path=self.data_path,
            length=self.length,
            transforms=self.transforms,
        )

        patients_idx = np.argwhere(
            list(map(lambda a: a in self.train_patients, self.dataset.get_ptients()))
        ).squeeze()
        self.train_dataset = self.dataset.subset(indices=patients_idx)

        patients_idx = np.argwhere(
            list(map(lambda a: a in self.test_patients, self.dataset.get_ptients()))
        ).squeeze()
        self.val_dataset = self.dataset.subset(indices=patients_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count(),
        )

    def __repr__(self) -> str:
        return (
            "#############################################################\n"
            + f"Dataset                    : {self.dataset.__class__.__name__}\n"
            + f"# Total Samples            : {len(self.dataset)}\n"
            + f"# Train Patients           : {self.train_patients}\n"
            + f"# Train Samples            : {len(self.train_dataset)}\n"
            + f"# Left Train Samples       : {self.train_dataset.get_class_distribution()[0]}\n"
            + f"# Right Train Samples      : {self.train_dataset.get_class_distribution()[1]}\n"
            + f"# Validation patients      : {self.test_patients}\n"
            + f"# Validation Samples       : {len(self.val_dataset)}\n"
            + f"# Left Validation Samples  : {self.val_dataset.get_class_distribution()[0]}\n"
            + f"# Right Validation Samples : {self.val_dataset.get_class_distribution()[1]}\n"
            + "#############################################################\n"
        )

    @staticmethod
    def collate_fn(batch):
        imgs = {
            key: torch.vstack([item[0][key].unsqueeze(0) for item in batch])
            for key in batch[0][0].keys()
        }
        trgts = {}
        sample_item_label = batch[0][1]
        for label_key in sample_item_label.keys():
            if isinstance(sample_item_label[label_key], dict):
                trgts[label_key] = {
                    key: torch.vstack(
                        [item[1][label_key][key].squeeze() for item in batch]
                    )
                    for key in sample_item_label[label_key].keys()
                }
            # else:
            # trgts[label_key] = torch.vstack([item[1][label_key] for item in batch]).squeeze()

        return [imgs, trgts]


class PaseEEGBCIIV2aDataLit(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        channels_config: str,
        train_patients: List[int],
        test_patients: List[int],
        batch_size: int = 32,
        workers_config: str = None,
        transforms=[],
    ):
        super().__init__()
        self.data_path = data_path
        self.eeg_electrode_positions, _ = eeg_electrode_configs(channels_config)
        self.batch_size = batch_size
        self.train_patients = train_patients
        self.test_patients = test_patients

        worker_configs = read_json_config(workers_config)
        worker_transforms = transforms_from_worker_configs(worker_configs)

        transforms = list(map(instantiate_class, transforms))

        self.transforms = Compose(worker_transforms + transforms)
        print(self.transforms)

    def setup(self, stage: Optional[str] = None):
        self.dataset = BCI2aDataset(
            eeg_electrode_positions=self.eeg_electrode_positions,
            data_path=self.data_path,
            transforms=self.transforms,
        )

        patients_idx = np.argwhere(
            list(
                map(
                    lambda a: int(a[2]) in self.train_patients,
                    self.dataset.get_ptients(),
                )
            )
        ).squeeze()
        self.train_dataset = self.dataset.subset(indices=patients_idx)

        patients_idx = np.argwhere(
            list(
                map(
                    lambda a: int(a[2]) in self.test_patients,
                    self.dataset.get_ptients(),
                )
            )
        ).squeeze()
        self.val_dataset = self.dataset.subset(indices=patients_idx)

        print(self)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=os.cpu_count(),
        )

    def __repr__(self) -> str:
        return (
            "#############################################################\n"
            + f"Dataset                     : {self.dataset.__class__.__name__}\n"
            + f"# Total Samples             : {len(self.dataset)}\n"
            + f"# Train Patients            : {self.train_patients}\n"
            + f"# Train Samples             : {len(self.train_dataset)}\n"
            + f"# Left Train Samples        : {self.train_dataset.get_class_distribution()[1]}\n"
            + f"# Right Train Samples       : {self.train_dataset.get_class_distribution()[2]}\n"
            + f"# Foot Train Samples        : {self.train_dataset.get_class_distribution()[3]}\n"
            + f"# Tongue Train Samples      : {self.train_dataset.get_class_distribution()[4]}\n\n"
            + f"# Validation patients       : {self.test_patients}\n"
            + f"# Validation Samples        : {len(self.val_dataset)}\n"
            + f"# Left Validation Samples   : {self.val_dataset.get_class_distribution()[1]}\n"
            + f"# Right Validation Samples  : {self.val_dataset.get_class_distribution()[2]}\n"
            + f"# Foot Validation Samples   : {self.val_dataset.get_class_distribution()[3]}\n"
            + f"# Tongue Validation Samples : {self.val_dataset.get_class_distribution()[4]}\n"
            + "#############################################################\n"
        )

    @staticmethod
    def collate_fn(batch):
        imgs = {
            key: torch.vstack([item[0][key].unsqueeze(0) for item in batch])
            for key in batch[0][0].keys()
        }
        trgts = {}
        sample_item_label = batch[0][1]
        for label_key in sample_item_label.keys():
            if isinstance(sample_item_label[label_key], dict):
                trgts[label_key] = {
                    key: torch.vstack(
                        [item[1][label_key][key].squeeze() for item in batch]
                    )
                    for key in sample_item_label[label_key].keys()
                }
            # else:
            # trgts[label_key] = torch.vstack([item[1][label_key] for item in batch]).squeeze()

        return [imgs, trgts]
