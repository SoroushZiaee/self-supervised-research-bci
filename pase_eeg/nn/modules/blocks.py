import torch
from torch import nn
from torch.nn import functional as F
from .conv import SincConvFast
from .utils import get_activation, get_norm_layer

from typing import Optional, List, Tuple, Union, Callable
from torch import Tensor
from torch.nn.common_types import _size_2_t


def forward_norm(x, norm_layer):
    if norm_layer is not None:
        if isinstance(norm_layer, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = norm_layer(x)
        if isinstance(norm_layer, nn.LayerNorm):
            x = x.transpose(1, 2)
        return x
    else:
        return x


def forward_activation(activation, tensor):
    if activation == "glu":
        # split tensor in two in channels dim
        z, g = torch.chunk(tensor, 2, dim=1)
        y = z * torch.sigmoid(g)
        return y
    else:
        return activation(tensor)


def get_padding(kwidth, dilation):
    return (kwidth // 2) * dilation


class NeuralBlock(nn.Module):
    def __init__(self, name="NeuralBlock"):
        super().__init__()
        self.name = name

    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/5
    def describe_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        print("-" * 10)
        print(self)
        print("Num params: ", pp)
        print("-" * 10)
        return pp


class SincBlock(NeuralBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t,
        padding: Union[str, _size_2_t] = "SAME",
        dilation: _size_2_t = 1,
        pad_mode: str = "reflect",
        activation=None,
        norm_type=None,
        sr=16000,
        name="SincBlock",
    ):
        super().__init__(name=name)
        if activation is not None and activation == "glu":
            Wfmaps = 2 * out_channels
        else:
            Wfmaps = out_channels
        self.num_inputs = in_channels
        self.fmaps = out_channels
        self.kwidth = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad_mode = pad_mode
        # only one-channel signal can be analyzed
        assert in_channels == 1, in_channels

        self.conv = SincConvFast(
            1,
            Wfmaps,
            kernel_size,
            sample_rate=sr,
            padding=padding,
            stride=stride,
            padding_mode=pad_mode,
        )
        # self.conv = nn.Conv1d(
        #     1,
        #     Wfmaps,
        #     kernel_size,
        #     padding=padding,
        #     stride=stride,
        #     padding_mode=pad_mode,
        # )
        if not (norm_type == "snorm"):
            self.norm = get_norm_layer(norm_type, self.conv, Wfmaps)
        else:
            self.norm = None
        self.act = get_activation(activation, out_channels)

    def forward(self, x):
        h = self.conv(x)

        if self.norm is not None:
            h = forward_norm(h, self.norm)

        h = forward_activation(self.act, h)
        # h = self.act(h)
        return h


class ResBlock(NeuralBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        padding: Union[str, _size_2_t] = "SAME",
        dilations: _size_2_t = 1,
        downsample: int = 1,
        pad_mode: str = "constant",
        activation=None,
        norm_type=None,
        hidden_channels=None,
        name="ResBlock",
    ):
        super().__init__(name=name)
        if hidden_channels is None:
            hidden_channels = out_channels

        self.num_inputs = in_channels
        self.fmaps = hidden_channels
        self.emb_dim = out_channels
        self.kwidth = kernel_size
        downscale = 1.0 / downsample
        self.downscale = downscale
        # stride is ignored for no downsampling is
        # possible in ResBlock
        self.stride = 1
        # self.dilation = dilation
        dilation = dilations[0]
        self.conv1 = nn.Conv1d(
            self.num_inputs,
            self.fmaps,
            kernel_size,
            dilation=dilation,
            padding=get_padding(self.kwidth, dilation),
        )
        self.norm1 = get_norm_layer(norm_type, self.conv1, self.fmaps)
        self.act1 = get_activation(activation, self.fmaps)
        dilation = dilations[1]
        self.conv2 = nn.Conv1d(
            self.fmaps,
            self.emb_dim,
            kernel_size,
            dilation=dilation,
            padding=get_padding(self.kwidth, dilation),
        )
        # assert self.norm2 is not None
        self.norm2 = get_norm_layer(norm_type, self.conv2, self.emb_dim)
        self.act2 = get_activation(activation, self.emb_dim)
        if self.num_inputs != self.emb_dim:
            # build projection layer
            self.resproj = nn.Conv1d(self.num_inputs, self.emb_dim, 1)

    def forward(self, x):
        """
        # compute pad factor
        if self.kwidth % 2 == 0:
            if self.dilation > 1:
                raise ValueError('Not supported dilation with even kwdith')
            P = (self.kwidth // 2 - 1,
                 self.kwidth // 2)
        else:
            pad = (self.kwidth // 2) * (self.dilation - 1) + \
                    (self.kwidth // 2)
            P = (pad, pad)
        """
        if self.downscale < 1:
            x = F.interpolate(x, scale_factor=self.downscale)

        identity = x
        # x = F.pad(x, P, mode=self.pad_mode)
        x = self.conv1(x)
        x = forward_norm(x, self.norm1)
        x = forward_activation(self.act1, x)
        x = self.conv2(x)
        x = forward_activation(self.act2, x)
        if hasattr(self, "resproj"):
            identity = self.resproj(identity)

        x = x + identity
        x = forward_norm(x, self.norm2)
        return x


class ChannelwiseFreqDomain(nn.Module):
    """ """

    def __init__(
        self,
        channel_positions,
        out_shape,
        name="WaveFe",
    ):
        super().__init__()
        self.channel_positions = channel_positions
        self.out_shape = out_shape

        self.sinc = SincBlock(
            in_channels=1,
            out_channels=64,
            kernel_size=101,
            stride=1,
            padding="valid",
            activation="prelu",
            norm_type="bnorm",
            sr=256,
        )
        self.blocks = nn.ModuleList()
        self.blocks.append(
            ResBlock(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding="valid",
                dilations=[1, 2],
                activation="prelu",
                norm_type="bnorm",
            )
        )

        self.blocks.append(
            ResBlock(
                in_channels=64,
                hidden_channels=64,
                out_channels=1,
                kernel_size=3,
                padding="valid",
                dilations=[1, 2],
                activation="prelu",
                norm_type="bnorm",
            )
        )

    def forward(self, batch_dict, device=None, mode=None):
        values = list(batch_dict.values())
        batch_size = values[0].size()[0]
        x = torch.vstack(values)

        x = self._forward_feature_extractor(x)

        x = self._flatten_by_position_2d(x, batch_size)

        return x

    def _forward_feature_extractor(self, x):
        x = self.sinc(x)
        # print(x.size())

        for block in self.blocks:
            x = block(x)
            # print(x.size())

        return x

    def _flatten_by_position_2d(self, x, batch_size):
        res = torch.zeros((batch_size, *self.out_shape, x.size()[-1]), device=x.device)
        for i, value in enumerate(self.channel_positions.values()):
            idx = self._position_to_index(value, self.out_shape)
            res[:, idx[0], idx[1], :] = x[
                i * batch_size : (i + 1) * batch_size, :, :
            ].squeeze()

        return res

    def _position_to_index(self, position, shape):
        center = (shape[0] // 2, shape[1] // 2)
        index = (center[0] - position[1], center[1] + position[0])
        return index


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = None,
    ) -> None:
        super(BasicBlock, self).__init__()

        self.dropout = dropout

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = norm_layer(planes)
        if dropout is not None:
            self.dropout1 = nn.Dropout2d(p=dropout)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = norm_layer(planes)
        if dropout is not None:
            self.dropout2 = nn.Dropout2d(p=dropout)
        # self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.dropout is not None:
            out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        if self.dropout is not None:
            out = self.dropout2(out)

        return out


class BlockCls2d(nn.Module):
    """ """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        name="Embedding-Classifier",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv_cls = nn.Conv2d(
            self.in_channels,
            self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_classes, self.num_classes)

    def forward(self, x, device=None, mode=None):
        x = self.conv_cls(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BlockEmb2d(nn.Module):
    """ """

    def __init__(
        self,
        in_channels: int,
        emb_dim: int,
        name="EEG-Embedder",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.ModuleList()
        self.blocks.append(
            BasicBlock(
                inplanes=self.inplanes,
                planes=self.inplanes,
                stride=1,
                base_width=64,
                dilation=1,
                dropout=0.2,
            )
        )

        self.blocks.append(
            BasicBlock(
                inplanes=self.inplanes,
                planes=self.inplanes,
                stride=1,
                base_width=64,
                dilation=1,
                dropout=0.2,
            )
        )

        self.conv_emb = nn.Conv2d(
            self.inplanes,
            self.emb_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x, device=None, mode=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for block in self.blocks:
            x = block(x)

        x = self.conv_emb(x)
        x = self.relu(x)
        return x


class DepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


class SeparableConv2D(nn.Module):
    """https://github.com/seungjunlee96/Depthwise-Separable-Convolution_Pytorch/blob/master/DepthwiseSeparableConvolution/DepthwiseSeparableConvolution.py"""

    def __init__(
        self,
        in_channels,
        out_channels,
        depth_multiplier=1,
        kernel_size=3,
        padding="valid",
        bias=False,
    ):
        super(SeparableConv2D, self).__init__()
        self.depthwise = DepthwiseConv2d(
            in_channels, depth_multiplier, kernel_size, padding=padding, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
