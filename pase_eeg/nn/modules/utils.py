from torch import nn
from torch.nn import functional as F
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.utils.weight_norm import weight_norm

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def get_norm_layer(norm_type, param=None, num_feats=None):
    if norm_type == "bnorm":
        return nn.BatchNorm1d(num_feats)
    elif norm_type == "snorm":
        spectral_norm(param)
        return None
    elif norm_type == "bsnorm":
        spectral_norm(param)
        return nn.BatchNorm1d(num_feats)
    elif norm_type == "lnorm":
        return nn.LayerNorm(num_feats)
    elif norm_type == "wnorm":
        weight_norm(param)
        return None
    elif norm_type == "inorm":
        return nn.InstanceNorm1d(num_feats, affine=False)
    elif norm_type == "affinorm":
        return nn.InstanceNorm1d(num_feats, affine=True)
    elif norm_type is None:
        return None
    else:
        raise TypeError("Unrecognized norm type: ", norm_type)


def get_activation(activation, params, init=0):
    if activation == "prelu" or activation is None:
        return nn.PReLU(params, init=init)
    if isinstance(activation, str):
        return getattr(nn, activation)()
    else:
        return activation
