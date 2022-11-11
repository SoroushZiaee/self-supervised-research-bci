from pase_eeg.nn.models.simple_classifier import EEGCls
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY

from pase_eeg.lit_modules.crossval_classifier_lit import BaseKFoldDataModule, KFoldLoop

from pase_eeg.lit_modules import simple_classifier_lit, pase_lit


MODEL_REGISTRY.register_classes(simple_classifier_lit, LightningModule)
DATAMODULE_REGISTRY.register_classes(simple_classifier_lit, LightningDataModule)

# MODEL_REGISTRY.register_classes(pase_lit, LightningModule)
# DATAMODULE_REGISTRY.register_classes(pase_lit, LightningDataModule)

if __name__ == "__main__":
    # PYTHONPATH=./:../../workspace/pytorch-lightning-snippets/ python pase_eeg/cli/crossval.py --config=configs/simple_classifier/config_trainer_base.yaml --config=configs/simple_classifier/crossval/config_BCIIV2a.yaml --trainer.gpus=1
    cli = LightningCLI(run=False)

    internal_fit_loop = cli.trainer.fit_loop
    cli.trainer.fit_loop = KFoldLoop()
    cli.trainer.fit_loop.connect(internal_fit_loop)
    cli.trainer.fit(cli.model, cli.datamodule)
