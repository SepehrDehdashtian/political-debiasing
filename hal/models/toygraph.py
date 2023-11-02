# folktables.py

from torch import nn

__all__ = ['EncToyGraph', 'TgtToyGraph', 'AdvToyGraph']


class EncToyGraph(nn.Module):  # hdl:128 r:1
    def __init__(self, ndim, r, hdl):
        super().__init__()

        self.enc_equal = nn.Sequential(
            nn.Linear(ndim, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl // 2),
            nn.PReLU(),
            nn.Linear(hdl // 2, r)
        )

    def forward(self, x, y=None):
        z = self.enc_equal(x)
        return z


class TgtToyGraph(nn.Module):  # hdl:128
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.dec_equal2 = nn.Sequential(
            nn.Linear(r, nout)
        )

    def forward(self, z, y=None):
        out = self.dec_equal2(z)
        return out



class AdvToyGraph(nn.Module):
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.adv = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

    def forward(self, z, y=None):
        out = self.adv(z)
        return out