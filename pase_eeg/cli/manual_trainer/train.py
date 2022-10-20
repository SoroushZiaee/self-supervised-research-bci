import os
import sys
import torch
import numpy as np
import random

import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

log = logging.getLogger(__name__)


def init():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


@hydra.main(
    version_base=None,
    config_path="../../../configs/hydra/simple_classifier",
    config_name="config",
)
def run(config):
    log.info(os.getcwd())
    init()
    trainer = instantiate(config.trainer)
    trainer.fit()


if __name__ == "__main__":
    log.info("Start Running the Program..")
    sys.argv.append('hydra.job.chdir=True')
    run()
