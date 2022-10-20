import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.blocks import DepthwiseConv2d, SeparableConv2D


class EEGNet(nn.Module):
    """
    Note: Use this class carefully. It is specificaly adapted with BCI Comp. IV 2a
        (22 channels, sample rate of 250, 3 seconds). So it's not a general class.
        it is developed only for experiment purposes.
    Implementation borrowed from:
        https://github.com/aliasvishnu/EEGNet
    There is also an official implementation with tensorflow here:
        https://github.com/vlawhern/arl-eegmodels
    Paper:
        https://arxiv.org/abs/1611.08024
    """

    def __init__(
        self,
        name="EEGNet",
    ):
        super(EEGNet, self).__init__()
        self.T = 250

        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 22), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 11))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 2)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 2))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 250 timepoints.
        self.fc1 = nn.Linear(4 * 4 * 99, 4)

    def forward(self, x, device=None):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        if self.training:
            x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)

        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        if self.training:
            x = F.dropout(x, 0.5)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        if self.training:
            x = F.dropout(x, 0.5)
        x = self.pooling3(x)

        # FC Layer
        # x = x.view(-1, 4*2*7)
        x = x.reshape((-1, 4 * 4 * 99))
        # x = torch.sigmoid(self.fc1(x))

        x = F.softmax(self.fc1(x), dim=1)
        return x


class EEGNetv2Emb(nn.Module):
    def __init__(
        self,
        emb_dim=128,
        # Change from 22 -> 1
        channels=1,
        dropout_rate=0.5,
        kernel_length=32,
        F1=8,
        D=2,
        F2=16,
    ):
        super(EEGNetv2Emb, self).__init__()
        self.dropout_rate = dropout_rate

        # Layer 1
        self.conv1 = nn.Conv2d(1, F1, (kernel_length, 1), padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, False)
        self.dwconv2 = DepthwiseConv2d(
            in_channels=F1,
            depth_multiplier=D,
            kernel_size=(1, channels),
            stride=1,
            padding="valid",
            bias=False,
        )

        self.batchnorm2 = nn.BatchNorm2d(2 * F1, False)
        # act elu
        self.pooling1 = nn.AvgPool2d((4, 1))
        # dropout

        # Layer 2
        self.sepconv2 = SeparableConv2D(2 * F1, F2, 1, (16, 1), padding="same")
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
        # elu
        self.pooling2 = nn.AvgPool2d((8, 1))
        # dropout

        # FC Layer
        self.fc1 = nn.Linear(16 * 24 * 1, emb_dim)

    def _forward_emb(self, x, device=None):
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.dwconv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        if self.training:
            x = F.dropout(x, self.dropout_rate)

        x = self.sepconv2(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        if self.training:
            x = F.dropout(x, self.dropout_rate)

        # FC Layer
        x = x.reshape((-1, 16 * 24 * 1))
        return x

    def forward(self, x, device=None):
        x = self._forward_emb(x)
        x = self.fc1(x)
        return x


class EEGNetv2(nn.Module):
    """
    Note: Use this class carefully. It is specificaly adapted with BCI Comp. IV 2a
        (22 channels, sample rate of 250, 3 seconds). So it's not a general class.
        it is developed only for experiment purposes.
    Implemented based on keras/tensorflow implementation found here:
        https://github.com/vlawhern/arl-eegmodels
    Paper:
        https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
    """

    def __init__(
        self,
        num_classes=4,
        channels=22,
        dropout_rate=0.5,
        kernel_length=32,
        F1=8,
        D=2,
        F2=16,
    ):
        super(EEGNetv2, self).__init__()
        self.dropout_rate = dropout_rate

        # Layer 1
        self.conv1 = nn.Conv2d(1, F1, (kernel_length, 1), padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, False)
        self.dwconv2 = DepthwiseConv2d(
            in_channels=F1,
            depth_multiplier=D,
            kernel_size=(1, channels),
            stride=1,
            padding="valid",
            bias=False,
        )

        self.batchnorm2 = nn.BatchNorm2d(2 * F1, False)
        # act elu
        self.pooling1 = nn.AvgPool2d((4, 1))
        # dropout

        # Layer 2
        self.sepconv2 = SeparableConv2D(2 * F1, F2, 1, (16, 1), padding="same")
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
        # elu
        self.pooling2 = nn.AvgPool2d((8, 1))
        # dropout

        # FC Layer
        self.fc1 = nn.Linear(16 * 24 * 1, 4)

    def _forward_emb(self, x, device=None):
        # Layer 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.dwconv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        if self.training:
            x = F.dropout(x, self.dropout_rate)

        x = self.sepconv2(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        if self.training:
            x = F.dropout(x, self.dropout_rate)

        # FC Layer
        x = x.reshape((-1, 16 * 24 * 1))
        return x

    def forward(self, x, device=None):
        x = self._forward_emb(x)
        x = F.softmax(self.fc1(x), dim=1)
        return x


class MBEEGNetv2(nn.Module):
    """
    Paper:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8773854/
    """

    def __init__(
        self,
    ):
        super(MBEEGNetv2, self).__init__()
        self.mbeegnet1 = EEGNetv2(
            num_classes=4,
            channels=22,
            dropout_rate=0,
            kernel_length=16,
            F1=4,
            D=2,
            F2=16,
        )
        self.mbeegnet2 = EEGNetv2(
            num_classes=4,
            channels=22,
            dropout_rate=0.1,
            kernel_length=32,
            F1=8,
            D=2,
            F2=16,
        )
        self.mbeegnet3 = EEGNetv2(
            num_classes=4,
            channels=22,
            dropout_rate=0.2,
            kernel_length=64,
            F1=16,
            D=2,
            F2=16,
        )

        # FC Layer
        self.fc1 = nn.Linear(16 * 24 * 3, 4)

    def forward(self, x, device=None):
        x = torch.cat(
            (
                self.mbeegnet1._forward_emb(x),
                self.mbeegnet2._forward_emb(x),
                self.mbeegnet3._forward_emb(x),
            ),
            dim=1,
        )
        x = F.softmax(self.fc1(x), dim=1)
        return x


class MBEEGNetv2Emb(nn.Module):
    def __init__(
        self,
        emb_dim,
    ):
        super(MBEEGNetv2Emb, self).__init__()
        self.mbeegnet1 = EEGNetv2(
            num_classes=4,
            # Change the channel from 22 -> 1
            channels=1,
            dropout_rate=0,
            kernel_length=16,
            F1=4,
            D=2,
            F2=16,
        )
        self.mbeegnet2 = EEGNetv2(
            num_classes=4,
            # Change the channel from 22 -> 1
            channels=1,
            dropout_rate=0.1,
            kernel_length=32,
            F1=8,
            D=2,
            F2=16,
        )
        self.mbeegnet3 = EEGNetv2(
            num_classes=4,
            # Change the channel from 22 -> 1
            channels=1,
            dropout_rate=0.2,
            kernel_length=64,
            F1=16,
            D=2,
            F2=16,
        )

        # FC Layer
        self.fc00 = nn.Linear(16 * 24 * 3, 1024)
        self.fc01 = nn.Linear(1024, 512)
        self.fc1 = nn.Linear(512, emb_dim)

    def forward(self, x, device=None):
        x = torch.cat(
            (
                self.mbeegnet1._forward_emb(x),
                self.mbeegnet2._forward_emb(x),
                self.mbeegnet3._forward_emb(x),
            ),
            dim=1,
        )
        x = F.tanh(self.fc1(F.relu(self.fc01(F.relu(self.fc00(x))))))
        return x
