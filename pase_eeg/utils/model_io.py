try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import re
import torch
from torch import nn

from collections import OrderedDict
from typing import Dict


def get_map_location():
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    return map_location


def merge_state_dicts(model_std, pretrained_std):
    merged_dict = {}
    for k, v in model_std.items():
        if k in pretrained_std and v.size() == pretrained_std[k].size():
            merged_dict[k] = pretrained_std[k]
        else:
            if k in pretrained_std:
                print(
                    f"not merged {k} with size {v.size()} {pretrained_std.get(k, None).size()}"
                )
            else:
                print(f"not merged {k} with size {v.size()} {None}")
            merged_dict[k] = v

    return merged_dict


def merge_state_dicts_by_key_mapping(
    model_std, pretrained_std, key_mappings, ignore_keys
):
    merged_dict = {}
    for k, v in model_std.items():
        if k in ignore_keys:
            merged_dict[k] = v
        elif k in key_mappings.keys():
            merged_dict[k] = pretrained_std[key_mappings[k]]
        elif k in pretrained_std and v.size() == pretrained_std[k].size():
            merged_dict[k] = pretrained_std[k]
        else:
            if k in pretrained_std:
                print(
                    f"not merged {k} with size {v.size()} {pretrained_std.get(k, None).size()}"
                )
            else:
                print(f"not merged {k} with size {v.size()} {None}")
            merged_dict[k] = v

    return merged_dict


def load_chpt(model, source, **kwargs):
    if isinstance(source, OrderedDict):
        state_dict = source
    elif isinstance(source, str):
        state_dict = torch.load(source)

    std = model.state_dict()

    state_dict = merge_state_dicts_manual(std, state_dict, **kwargs)
    model.load_state_dict(state_dict)

    return model
