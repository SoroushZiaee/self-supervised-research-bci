from typing import Dict, List, Tuple, Any
import json
from pase_eeg.data.transforms import (
    LabelToDict,
    WTE,
    WPTE,
    PSD,
    Kurtosis,
    PTPAmplitude,
    Skewness,
)

from random import choice
from string import ascii_letters


def eeg_electrode_configs(
    conf_path: str,
) -> Tuple[Dict[str, Tuple[int, int]], Tuple[int, int]]:
    config = {}
    with open(conf_path, "r") as f:
        lines = f.readlines()
        lines = "".join(lines)
        exec(lines, config)
        return config["eeg_electrode_positions"], config["eeg_electrods_plane_shape"]


def read_json_config(
    conf_path: str,
) -> List[Dict[str, Any]]:
    with open(conf_path, "r") as f:
        configs = json.load(f)
        return configs


def transforms_from_worker_configs(worker_configs):
    transforms = []
    for worker in worker_configs:
        name = worker["name"]
        kwargs = worker.get("transform", {})
        if name == "wte":
            transforms.append(WTE(**kwargs))
        elif name == "wpte":
            transforms.append(WPTE(**kwargs))
        elif name == "psd":
            transforms.append(PSD(**kwargs))
        elif name == "kurtosis":
            transforms.append(Kurtosis(**kwargs))
        elif name == "ptp-amp":
            transforms.append(PTPAmplitude(**kwargs))
        elif name == "skewness":
            transforms.append(Skewness(**kwargs))

    if len(transforms) > 0:
        transforms.insert(0, LabelToDict())

    return transforms


def instantiate_class(init: Dict[str, Any]) -> Any:
    """Instantiates a class with the given args and init.
    from:
        https://github.com/PyTorchLightning/pytorch-lightning/blob/c278802b64c10b838ab94d3edc862dd4df65a0a8/pytorch_lightning/utilities/cli.py#L895

    Args:
        init: Dict of the form {"class_path":...,"init_args":...}.
    Returns:
        The instantiated class object.
    """
    kwargs = init.get("init_args", {})
    class_module, class_name = init["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(**kwargs)


def random_string(length=12):
    return "".join(choice(ascii_letters) for i in range(length))
