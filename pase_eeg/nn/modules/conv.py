import torch
from torch import nn
from torch.nn import functional as F

import math
import numpy as np

from typing import Optional, List, Tuple, Union
from torch import Tensor
from torch.nn.common_types import _size_2_t


# Modified from https://github.com/mravanelli/SincNet
class SincConvFast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.

    Usage
    -----
    See `torch.nn.Conv1d`

    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz: int) -> int:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel: int) -> int:
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = "VALID",
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "reflect",
        sample_rate: int = 16000,
        min_low_hz: int = 1,
        min_band_hz: int = 5,
    ) -> None:
        super(SincConvFast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = (
                "SincConv only support one input channel (here, in_channels = {%i})"
                % (in_channels)
            )
            raise ValueError(msg)

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            # TODO: needs verbosity log
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.pad_mode = padding_mode
        self.dilation = dilation

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 1
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(
            self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1
        )
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(
            torch.Tensor(hz[:-1]).view(-1, 1), requires_grad=True
        )

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(
            torch.Tensor(np.diff(hz)).view(-1, 1), requires_grad=True
        )

        # self.prev_low_hz_ = self.low_hz_.detach().clone()
        # self.prev_band_hz_ = self.band_hz_.detach().clone()

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )  # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        # (kernel_size, 1)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = (
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms: Tensor) -> Tensor:
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.

        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)

        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET).
        # I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )

        band_pass = band_pass / (2 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        x = waveforms

        if self.padding == "SAME":
            if self.stride > 1:
                x_p = F.pad(
                    x,
                    (self.kernel_size // 2 - 1, self.kernel_size // 2),
                    mode=self.pad_mode,
                )
            else:
                x_p = F.pad(
                    x,
                    (self.kernel_size // 2, self.kernel_size // 2),
                    mode=self.pad_mode,
                )
        else:
            x_p = x

        # if torch.equal(self.prev_low_hz_.to(self.low_hz_.device), self.low_hz_) or torch.equal(self.prev_band_hz_.to(self.low_hz_.device), self.band_hz_):
        #     print("Sinc Weights didn't get updated...")
        #     if self.low_hz_.grad is not None:
        #         print(torch.max(self.low_hz_.grad))
        #         print(torch.min(self.low_hz_.grad))
        # else:
        #     print(self.low_hz_.detach().clone())
        #     self.prev_low_hz_ = self.low_hz_.detach().clone()
        #     self.prev_band_hz_ = self.band_hz_.detach().clone()

        return F.conv1d(
            x_p,
            self.filters,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )
