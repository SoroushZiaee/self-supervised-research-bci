from torch import nn
from ..modules.blocks import ChannelwiseFreqDomain, BlockEmb2d


class EEGEmb(nn.Module):
    """ """

    def __init__(
        self,
        channel_positions,
        channels_plane_shape=(11, 13),
        emb_dim=256,
        name="EEG-Classifier",
    ):
        super().__init__()
        self.channel_positions = channel_positions
        self.channels_plane_shape = channels_plane_shape

        self.signal_block = ChannelwiseFreqDomain(
            channel_positions=self.channel_positions,
            out_shape=self.channels_plane_shape,
        )
        self.spacial_block = BlockEmb2d(in_channels=651, emb_dim=emb_dim)

    def forward(self, batch_dict, device=None, mode=None):
        x = self.signal_block(batch_dict, device=device)
        x = x.permute(0, 3, 1, 2)
        x = self.spacial_block(x)
        return x
