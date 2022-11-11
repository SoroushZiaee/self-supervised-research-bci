import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Binomial
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.utils.weight_norm import weight_norm
import numpy as np


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


class PatternedDropout(nn.Module):
    def __init__(
        self,
        emb_size,
        p=0.5,
        dropout_mode=["fixed_rand"],
        ratio_fixed=None,
        range_fixed=None,
        drop_whole_channels=False,
    ):
        """Applies a fixed pattern of dropout for the whole training
        session (i.e applies different only among pre-specified dimensions)
        """
        super(PatternedDropout, self).__init__()

        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

        if self.p > 0:

            d_modes = ["std", "fixed_rand", "fixed_given"]
            assert (
                dropout_mode in d_modes
            ), "Expected dropout mode in {}, got {}".format(d_modes, dropout_mode)
            self.drop_whole_channels = drop_whole_channels
            self.dropout_fixed = False

            if dropout_mode == "fixed_rand":

                self.dropout_fixed = True
                assert (
                    ratio_fixed is not None
                ), "{} needs 'ratio_fixed' arg set.".format(dropout_mode)
                if ratio_fixed <= 0 or ratio_fixed > 1:
                    raise ValueError(
                        "{} mode needs 'ratio_fixed' to be"
                        " set and in (0, 1) range, got {}".format(
                            dropout_mode, ratio_fixed
                        )
                    )
                self.ratio_fixed = ratio_fixed
                self.dropped_dimsize = int(emb_size - emb_size * ratio_fixed)
                tot_idx = np.arange(emb_size)
                sel_idx = np.sort(
                    np.random.choice(tot_idx, size=self.dropped_dimsize, replace=False)
                )

            elif dropout_mode == "fixed_given":

                self.dropout_fixed = True
                if (
                    range_fixed is None
                    or not isinstance(range_fixed, str)
                    or len(range_fixed.split(":")) < 2
                ):
                    raise ValueError(
                        "{} mode needs 'range_dropped' to be"
                        " set (i.e. 10:20)".format(dropout_mode)
                    )
                rng = range_fixed.split(":")
                beg = int(rng[0])
                end = int(rng[1])
                assert beg < end and end <= emb_size, "Incorrect range {}".format(
                    range_fixed
                )
                self.dropped_dimsize = int(emb_size - (end - beg))
                tot_idx = np.arange(emb_size)
                fixed_idx = np.arange(beg, end, 1)
                sel_idx = np.setdiff1d(tot_idx, fixed_idx, assume_unique=True)

            if self.dropout_fixed:
                assert (
                    len(sel_idx) > 0
                ), "Asked for fixed dropout, but sel_idx {}".format(sel_idx)
                print(
                    "Enabled dropout mode: {}. p={}, drop channels {}. Selected indices to apply dropout are: {}".format(
                        dropout_mode, self.p, drop_whole_channels, sel_idx
                    )
                )
                self.dindexes = torch.LongTensor(sel_idx)
                self.p = p
                self.p_scale = 1.0 / (1.0 - self.p)
            else:
                # notice, it is up to the user to make sure experiments between
                # fixed dropout and regular one are comparabe w.r.t p (i.e. for
                # fixed mode we only keep droping a subset of units (i.e. 50%),
                # thus p is effectively lower when compared to regular dropout)
                self.p = p
                print("Using std dropout with p={}".format(self.p))
        else:
            print("Dropout at the inputs disabled, as p={}".format(self.p))

    def forward(self, x):

        if self.p == 0 or not self.training:
            return x

        if self.dropout_fixed and self.training:
            self.dindexes = self.dindexes.to(x.device)
            assert (
                len(x.size()) == 3
            ), "Expected to get 3 dimensional tensor, got {}".format(len(x.size()))
            bsize, emb_size, tsize = x.size()
            # print (bsize, esize, tsize)
            if self.drop_whole_channels:
                batch_mask = torch.full(
                    size=(bsize, emb_size), fill_value=1.0, device=x.device
                )
                probs = torch.full(
                    size=(bsize, self.dropped_dimsize), fill_value=1.0 - self.p
                )
                b = Binomial(total_count=1, probs=probs)
                mask = b.sample()
                mask = mask.to(x.device)
                batch_mask[:, self.dindexes] *= mask * self.p_scale
                # print ('mask dc', mask)
                # print ('maks dcv', mask.view(bsize, self.dropped_dimsize, -1))
                # x[:,self.dindexes,:] = x[:,self.dindexes,:].clone() * self.p_scale\
                #                         * mask.view(bsize, self.dropped_dimsize, -1)
                x = x * batch_mask.view(bsize, emb_size, -1)
            else:
                batch_mask = torch.ones_like(x, device=x.device)
                probs = torch.full(
                    size=(bsize, self.dropped_dimsize, tsize), fill_value=1.0 - self.p
                )
                b = Binomial(total_count=1, probs=probs)
                mask = b.sample()
                mask = mask.to(x.device)
                batch_mask[:, self.dindexes, :] *= mask * self.p_scale
                x = x * batch_mask
                # xx = x.data.clone()
                # x[:,self.dindexes,:] = x[:,self.dindexes,:].clone() * mask * self.p_scale
            return x
        else:
            return F.dropout(x, p=self.p, training=self.training)


class MLPBlock(NeuralBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout=0,
        context=1,
        name="MLPBlock",
    ):
        super().__init__(name=name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = nn.Conv2d(in_channels, out_channels, context, padding=context // 2)

        self.act = nn.PReLU(out_channels)
        self.dout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.W(x)
        x = self.act(x)
        x = self.dout(x)
        return x
