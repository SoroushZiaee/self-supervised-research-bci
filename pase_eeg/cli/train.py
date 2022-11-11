from pase_eeg.nn.models.simple_classifier import EEGCls
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY

from pase_eeg.lit_modules import simple_classifier_lit, pase_lit


MODEL_REGISTRY.register_classes(simple_classifier_lit, LightningModule)
DATAMODULE_REGISTRY.register_classes(simple_classifier_lit, LightningDataModule)

MODEL_REGISTRY.register_classes(pase_lit, LightningModule)
DATAMODULE_REGISTRY.register_classes(pase_lit, LightningDataModule)

if __name__ == "__main__":
    # PYTHONPATH=./ python pase_eeg/cli/train.py fit --model=EEGClsLit --data=EEGSynthetichDataLit --print_config
    # PYTHONPATH=./ python pase_eeg/cli/train.py fit --config=configs/simple_classifier/config_chb_mit.yaml --trainer.gpus=1
    cli = LightningCLI()
