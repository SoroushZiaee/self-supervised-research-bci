from torch.utils.tensorboard import SummaryWriter
from torch import Tensor

from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
import os
import re


class TensorBoardManualLogger(object):
    def __init__(self, save_dir: str = "/experiments/pase_eeg/manual_logs/"):
        # organize the folders
        self.save_dir = save_dir
        self._fix_prefolder()
        # Create the log directory
        os.makedirs(self.save_dir, exist_ok=True)

        self.experiment = SummaryWriter(log_dir=self.save_dir)

    def _fix_prefolder(self):
        def extract_num(string: str):
            return int(re.findall(r"version_(\d+)", string)[0])

        log_dirs = [
            item for item in os.listdir(self.save_dir) if item.startswith("version_")
        ]
        log_dirs.sort(key=lambda x: extract_num(x), reverse=True)

        if log_dirs != []:
            last_version = extract_num(log_dirs[0])

            print(f"{last_version = }")

            self.save_dir = os.path.join(
                self.save_dir, "version_{:02d}/".format(last_version + 1)
            )

        else:
            self.save_dir = os.path.join(self.save_dir, "version_{:02d}/".format(0))

    def log(self, metrics, step: Optional[int] = None) -> None:

        for k, v in metrics.items():
            if isinstance(v, Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.experiment.add_scalars(k, v, step)

            else:
                try:
                    self.experiment.add_scalar(k, v, step)
                # todo: specify the possible exception
                except Exception as ex:
                    m = f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    raise ValueError(m) from ex
