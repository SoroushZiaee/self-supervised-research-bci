from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
import os
import numpy as np

from torch import Tensor
import torch
from torch import nn
from torch.nn import functional as F

from torchmetrics.functional import f1_score, accuracy, cohen_kappa
from pytorch_lightning import LightningModule, LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from .utils import eeg_electrode_configs, instantiate_class

from ..nn.models.simple_classifier import EEGCls
from ..nn.models.eegnet import EEGNet, EEGNetv2, MBEEGNetv2
from ..data.synthetic_dataset import EEGSyntheticDataset
from ..data.chb_mit import CHBMITDataset
from ..data.LEE import LEEDataset
from ..data.Klinik import KlinikDataset
from ..data.bci import BCI2aDataset
from ..data.transforms import Compose

from ..utils.lr_scheduler import GradualWarmupScheduler
from ..utils.model_io import merge_state_dicts

import wandb


class EEGClsLit(LightningModule):
    def __init__(
        self,
        channels_config: str,
        num_classes: int,
        emb_dim=256,
        learning_rate: float = 3e-4,
        min_learning_rate: float = 1e-7,
        pretrained_backend_weights_path: str = None,
    ) -> None:

        super().__init__()
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate

        eeg_electrode_positions, eeg_electrods_plane_shape = eeg_electrode_configs(
            channels_config
        )

        self.model = EEGCls(
            channel_positions=eeg_electrode_positions,
            channels_plane_shape=eeg_electrods_plane_shape,
            emb_dim=emb_dim,
            num_classes=num_classes,
        )

        if pretrained_backend_weights_path is not None:
            self.model = self._load_weigths(self.model, pretrained_backend_weights_path)

    def _load_weigths(self, model, path):
        checkpoint = torch.load(path)
        pretrained_state_dict = {
            key.replace("model.", ""): value
            for key, value in checkpoint["state_dict"].items()
            if "model." in key
        }
        model_state__dict = self.model.state_dict()
        state_dict = merge_state_dicts(model_state__dict, pretrained_state_dict)

        model.load_state_dict(state_dict)

        return model

    def forward(self, x: Any) -> Any:
        x = self.model(x, device=self.device)
        return F.log_softmax(x, dim=1)

    def train_eval_step(
        self,
        batch: Tuple[Dict[str, Tensor], List[int]],
        name="",
        step: Optional[int] = None,
    ):
        x, y = batch
        logits = self(x)
        y = y.type(torch.int64).to(logits.device)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        f1_micro = f1_score(preds, y, average="micro")

        # Calling self.log will surface up scalars for you in TensorBoard
        # self.log(f"{name}_loss", loss, prog_bar=True)
        # self.log(f"{name}_acc", acc, prog_bar=True)
        # self.log(f"{name}_f1_micro", f1_micro, prog_bar=True)

        results = {
            f"loss/{name}": loss,
            f"acc/{name}": acc,
            f"f1_micro/{name}": f1_micro,
        }

        self.log(results, step)

        # Commnent the WANDB logger
        # if wandb.run is None:
        #     self.log(f"{name}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        #     self.log(f"{name}_f1(micro)", f1_micro, prog_bar=True)
        # else:
        #     wandb.log(
        #         {
        #             f"{name}_loss": loss,
        #             f"{name}_acc": acc,
        #             f"{name}_f1(micro)": f1_micro,
        #         }
        #     )

        return loss

    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], List[int]],
        batch_idx: int,
        num_epoch: int,
        optimizer_idx: int = None,
    ):
        return self.train_eval_step(batch=batch, name="train", step=num_epoch)

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], List[int]],
        batch_idx: int,
        num_epoch: int,
    ):
        return self.train_eval_step(batch=batch, name="eval", step=num_epoch)

    def configure_optimizers(self):
        params = list(self.parameters())
        optimizers = []
        schedulers = []

        optimizers.append(
            torch.optim.SGD(
                [params[0], params[1]],
                lr=self.learning_rate * 50,
                momentum=0.9,
                nesterov=True,
            )
        )
        optimizers.append(
            torch.optim.SGD(
                params[2:], lr=self.learning_rate, momentum=0.9, nesterov=True
            )
        )

        schedulers.append(
            GradualWarmupScheduler(
                optimizers[0],
                80,
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[0], T_max=220, eta_min=1e-9
                ),
            )
            # {
            #     "scheduler": GradualWarmupScheduler(
            #         optimizers[0],
            #         80,
            #         torch.optim.lr_scheduler.CosineAnnealingLR(
            #             optimizers[0], T_max=220, eta_min=1e-9
            #         ),
            #         # torch.optim.lr_scheduler.ReduceLROnPlateau(
            #         #     optimizer, min_lr=self.min_learning_rate
            #         # ),
            #     ),
            #     "interval": "epoch",
            #     "frequency": 1,
            #     "monitor": "train_loss",
            #     "strict": True,
            # }
        )
        schedulers.append(
            GradualWarmupScheduler(
                optimizers[1],
                80,
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[1], T_max=220, eta_min=1e-9
                ),
                # torch.optim.lr_scheduler.ReduceLROnPlateau(
                #     optimizer, min_lr=self.min_learning_rate
                # ),
            )
            # {
            #     "scheduler": GradualWarmupScheduler(
            #         optimizers[1],
            #         80,
            #         torch.optim.lr_scheduler.CosineAnnealingLR(
            #             optimizers[1], T_max=220, eta_min=1e-9
            #         ),
            #         # torch.optim.lr_scheduler.ReduceLROnPlateau(
            #         #     optimizer, min_lr=self.min_learning_rate
            #         # ),
            #     ),
            #     "interval": "epoch",
            #     "frequency": 1,
            #     "monitor": "train_loss",
            #     "strict": True,
            # }
        )
        return optimizers, schedulers

    def training_epoch_end(self, training_step_outputs):
        print("\n")

    def training_epoch_start(self, training_step_outputs):
        print("\n")


class EEGNetLit(LightningModule):
    """

    Parameters
    ----------
    model : str, optional
        name of the eegnet varient to use. one of 'eegnetv1', 'eegnetv2', 'mbeegnetv2'
        , by default "eegnetv1"
    learning_rate : float, optional
        , by default 3e-4
    min_learning_rate : float, optional
        , by default 1e-7
    decay_rate: float, optional
        weight decay used in Adam optimizer, by default 0.5
    pretrained_backend_weights_path : str, optional
        _description_, by default None
    """

    def __init__(
        self,
        model: str = "eegnetv1",
        learning_rate: float = 3e-4,
        min_learning_rate: float = 1e-7,
        decay_rate: float = 0.5,
        pretrained_backend_weights_path: str = None,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate

        if model == "eegnetv1":
            self.model = EEGNet()
        elif model == "eegnetv2":
            self.model = EEGNetv2(
                num_classes=4,
                channels=22,
                dropout_rate=0.1,
                kernel_length=32,
                F1=8,
                D=2,
                F2=16,
            )
        elif model == "mbeegnetv2":
            self.model = MBEEGNetv2()

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: Any) -> Any:
        return self.model(x, device=self.device)

    def train_eval_step(self, batch: Tuple[Dict[str, Tensor], List[int]], name=""):
        x, y = batch
        x = torch.permute(
            torch.vstack(list(map(lambda a: a.unsqueeze(0), x.values()))),
            (1, 2, 3, 0),
        )
        # wrap them in Variable
        y = y.type(torch.LongTensor).cuda(0)

        logits = self(x)
        y = y.type(torch.int64).to(logits.device)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        kappa = cohen_kappa(preds, y, num_classes=4)
        f1_micro = f1_score(preds, y, average="micro")

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log(f"{name}_loss", loss, prog_bar=True)
        if wandb.run is None:
            self.log(f"{name}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"{name}_kappa", kappa, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"{name}_f1(micro)", f1_micro, prog_bar=True)
        else:
            wandb.log(
                {
                    f"{name}_loss": loss,
                    f"{name}_acc": acc,
                    f"{name}_kappa": kappa,
                    f"{name}_f1(micro)": f1_micro,
                }
            )

        return loss

    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], List[int]],
        batch_idx: int,
        optimizer_idx: int=0,
    ):
        return self.train_eval_step(batch=batch, name="train")

    def validation_step(
        self, batch: Tuple[Dict[str, Tensor], List[int]], batch_idx: int
    ):
        return self.train_eval_step(batch=batch, name="eval")

    def configure_optimizers(self):
        params = list(self.parameters())
        optimizers = []
        schedulers = []
        optimizers.append(
            torch.optim.Adam(params, lr=self.learning_rate, weight_decay=1)
        )
        schedulers.append(
            {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizers[0], max_lr=0.001, steps_per_epoch=1, epochs=300
                ),
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


class EEGSynthetichDataLit(LightningDataModule):
    def __init__(self, channels_config: str, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

        self.eeg_electrode_positions, _ = eeg_electrode_configs(channels_config)

    def setup(self, stage: Optional[str] = None):
        self.dataset = EEGSyntheticDataset(
            eeg_electrode_positions=self.eeg_electrode_positions,
            transforms=ToTensor(device=torch.device("cuda:0")),
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
            collate_fn=self.dataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.subset(self.test_idx),
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
        )

    def __repr__(self) -> str:
        return (
            "#############################################################\n"
            + f"Dataset              : {self.dataset.__class__.__name__}\n"
            + f"# Total Samples      : {len(self.dataset)}\n"
            + f"# Train Samples      : {len(self.train_idx)}\n"
            + f"# Validation Samples : {len(self.test_idx)}\n"
            + "#############################################################\n"
        )


class EEGCHBMITDataLit(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        channels_config: str,
        patient_list: List[int],
        length: int,
        batch_size: int = 32,
        transforms=[],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.patient_list = patient_list
        self.length = length
        self.eeg_electrode_positions, _ = eeg_electrode_configs(channels_config)
        self.data_path = data_path

        self.transforms = Compose(list(map(instantiate_class, transforms)))
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
            collate_fn=self.dataset.collate_fn,
            drop_last=True,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.subset(self.test_idx),
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
            drop_last=True,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

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


class EEGLEEDataLit(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        channels_config: str,
        train_patients: List[int],
        test_patients: List[int],
        length: int,
        batch_size: int = 32,
        transforms=[],
    ):
        super().__init__()
        self.data_path = data_path
        self.eeg_electrode_positions, _ = eeg_electrode_configs(channels_config)
        self.batch_size = batch_size
        self.train_patients = train_patients
        self.test_patients = test_patients
        self.length = length

        self.transforms = Compose(list(map(instantiate_class, transforms)))
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
            collate_fn=self.dataset.collate_fn,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
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


class EEGKlinikDataLit(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        channels_config: str,
        patient_list: List[int],
        length: int,
        batch_size: int = 32,
        transforms=[],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.patient_list = patient_list
        self.length = length
        self.eeg_electrode_positions, _ = eeg_electrode_configs(channels_config)
        self.data_path = data_path

        self.transforms = Compose(list(map(instantiate_class, transforms)))
        print(self.transforms)

    def setup(self, stage: Optional[str] = None):
        self.dataset = KlinikDataset(
            eeg_electrode_positions=self.eeg_electrode_positions,
            data_path=self.data_path,
            length=self.length,
            transforms=self.transforms,
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
            collate_fn=self.dataset.collate_fn,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset.subset(self.test_idx),
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
            drop_last=True,
        )

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


class EEGBCIIV2aDataLit(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        channels_config: str,
        train_patients: List[int],
        test_patients: List[int],
        batch_size: int = 32,
        transforms=[],
        leave_one_out=False,
    ):
        super().__init__()
        self.data_path = data_path
        self.eeg_electrode_positions, _ = eeg_electrode_configs(channels_config)
        self.batch_size = batch_size
        self.train_patients = train_patients
        self.test_patients = test_patients

        self.leave_one_out = leave_one_out

        self.transforms = Compose(list(map(instantiate_class, transforms)))
        print(self.transforms)

    def setup(self, stage: Optional[str] = None):
        self.dataset = BCI2aDataset(
            eeg_electrode_positions=self.eeg_electrode_positions,
            data_path=self.data_path,
            transforms=self.transforms,
        )

        train_patients_idx = np.argwhere(
            list(
                map(
                    lambda a: int(a[2]) in self.train_patients,
                    self.dataset.get_ptients(),
                )
            )
        ).squeeze()

        test_patients_idx = np.argwhere(
            list(
                map(
                    lambda a: int(a[2]) in self.test_patients,
                    self.dataset.get_ptients(),
                )
            )
        ).squeeze()

        if not self.leave_one_out:
            ds = self.dataset.subset(indices=test_patients_idx)
            train_idx, test_idx, _, _ = train_test_split(
                test_patients_idx,
                ds.get_labels(),
                stratify=ds.get_labels(),
                test_size=0.2,
            )

            train_patients_idx = np.concatenate([train_patients_idx, train_idx])
            test_patients_idx = test_idx

        self.train_dataset = self.dataset.subset(indices=train_patients_idx)
        self.val_dataset = self.dataset.subset(indices=test_patients_idx)

        print(self)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.dataset.collate_fn,
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
