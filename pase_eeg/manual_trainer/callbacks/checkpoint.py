import os
from pathlib import Path
from typing import Union, List, Optional

import torch

import logging

log = logging.getLogger(__name__)


class ModelCheckpoint(object):
    def __init__(self):
        self.count = 0

    def on_epoch_end(self, trainer):
        self.count += 1
        checkpoint = {"state_dict": trainer.lit_model.model.state_dict()}
        checkpoint_path = os.path.join(os.getcwd(), f"{self.count}.ckpt")
        torch.save(checkpoint, checkpoint_path)
        log.info(f"Model Checkpoint save to {checkpoint_path}")
