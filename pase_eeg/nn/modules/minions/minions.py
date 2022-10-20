import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.modules import *
from .blocks import MLPBlock

import json
import random


def minion_maker(cfg):
    if isinstance(cfg, str):
        with open(cfg, "r") as f:
            cfg = json.load(f)
    print("=" * 50)
    print("name", cfg["name"])
    print("=" * 50)
    mtype = cfg.pop("type", "mlp")
    if mtype == "mlp":
        minion = MLPMinion(**cfg)
    elif mtype in ["wte", "wpte", "kurtosis", "ptp-amp", "skewness"]:
        minion = WTEMinion(**cfg)
    elif mtype in ["psd"]:
        minion = PSDMinion(**cfg)
    else:
        raise TypeError("Unrecognized minion type {}".format(mtype))
    return minion


# class MLPMinion(Model):
#     def __init__(
#         self,
#         in_shape,
#         out_shape,
#         dropout,
#         dropout_time=0.0,
#         hidden_size=256,
#         hidden_layers=2,
#         context=1,
#         loss_weight=1.0,
#         channels=None,
#         name="MLPMinion",
#     ):
#         super().__init__(name=name)
#         self.in_shape = in_shape
#         self.out_shape = out_shape
#         self.in_channels = in_shape[0]
#         self.out_channels = out_shape[0]

#         assert context % 2 != 0, context
#         self.context = context

#         self.dropout = dropout
#         self.dropout_time = dropout_time
#         self.hidden_size = hidden_size
#         self.hidden_layers = hidden_layers
#         self.loss_weight = loss_weight
#         self.channels = channels

#         self.criterion = nn.MSELoss()

#         self.blocks = nn.ModuleList()
#         ninp = self.in_channels
#         for _ in range(hidden_layers):
#             self.blocks.append(
#                 MLPBlock(
#                     ninp,
#                     hidden_size,
#                     dropout=dropout,
#                     context=context,
#                 )
#             )
#             ninp = hidden_size
#             context = 1

#         self.W = nn.ModuleDict()
#         for key in self.channels.keys():
#             self.W[key] = nn.Conv2d(
#                 ninp, self.out_channels, in_shape[1:], padding="valid"
#             )

#     def forward(self, x):
#         if self.dropout_time > 0 and self.context > 1:
#             mask = (
#                 (
#                     torch.FloatTensor(x.shape[0], x.shape[2]).to("cuda").uniform_()
#                     > self.dropout_time
#                 )
#                 .float()
#                 .unsqueeze(1)
#             )
#             x = x * mask

#         h = x
#         for block in self.blocks:
#             h = block(h)

#         logits = {}
#         for key in self.W.keys():
#             logits[key] = self.W[key](h).squeeze()

#             if len(logits[key].size()) < 2:
#                 logits[key] = logits[key].unsqueeze(1)

#         return logits

#     def loss(self, logits, targets):
#         total_loss = 0
#         for key in logits.keys():
#             total_loss += self.criterion(logits[key], targets[key])
#         return total_loss


class MLPMinion(Model):
    def __init__(
        self,
        in_shape,
        out_shape,
        dropout,
        dropout_time=0.0,
        hidden_size=256,
        hidden_layers=2,
        context=1,
        loss_weight=1.0,
        channels=None,
        name="MLPMinion",
    ):
        super().__init__(name=name)
        self.in_shape = in_shape
        self.out_shape = out_shape
        # self.in_channels = in_shape[0]
        # self.out_channels = out_shape[0]

        assert context % 2 != 0, context
        self.context = context

        self.dropout = dropout
        self.dropout_time = dropout_time
        # self.hidden_size = hidden_size
        # self.hidden_layers = hidden_layers
        self.loss_weight = loss_weight
        self.channels = channels

        self.criterion = nn.MSELoss()

        # self.blocks = nn.ModuleList()
        # ninp = self.in_channels
        # for _ in range(hidden_layers):
        #     self.blocks.append(
        #         MLPBlock(
        #             ninp,
        #             hidden_size,
        #             dropout=dropout,
        #             context=context,
        #         )
        #     )
        #     ninp = hidden_size
        #     context = 1

        self.W = nn.ModuleDict()
        for key in self.channels.keys():
            self.W[key] = nn.Linear(
                self.in_shape, self.out_shape
            )

    def forward(self, x):
        if self.dropout_time > 0 and self.context > 1:
            mask = (
                (
                    torch.FloatTensor(x.shape[0], x.shape[2]).to("cuda").uniform_()
                    > self.dropout_time
                )
                .float()
                .unsqueeze(1)
            )
            x = x * mask

        h = x
        # for block in self.blocks:
        #     h = block(h)

        logits = {}
        for key in self.W.keys():
            logits[key] = self.W[key](h).squeeze()

            if len(logits[key].size()) < 2:
                logits[key] = logits[key].unsqueeze(1)

        return logits

    def loss(self, logits, targets):
        total_loss = 0
        for key in logits.keys():
            total_loss += self.criterion(logits[key], targets[key])
        return total_loss


class WTEMinion(MLPMinion):
    def __init__(
        self,
        in_shape,
        out_shape,
        dropout,
        dropout_time=0.0,
        hidden_size=256,
        hidden_layers=2,
        context=1,
        loss_weight=1.0,
        channels=None,
        name="MLPMinion",
    ):
        super().__init__(
            in_shape,
            out_shape,
            dropout,
            dropout_time,
            hidden_size,
            hidden_layers,
            context,
            loss_weight,
            channels,
            name,
        )


class PSDMinion(MLPMinion):
    def __init__(
        self,
        in_shape,
        out_shape,
        dropout,
        dropout_time=0.0,
        hidden_size=256,
        hidden_layers=2,
        context=1,
        loss_weight=1.0,
        channels=None,
        name="MLPMinion",
    ):
        super().__init__(
            in_shape,
            out_shape,
            dropout,
            dropout_time,
            hidden_size,
            hidden_layers,
            context,
            loss_weight,
            channels,
            name,
        )
