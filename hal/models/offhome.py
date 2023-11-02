from torch import nn

__all__ = ['EncOffHome', 'TgtOffHome', 'AdvOffHome']


class EncOffHome(nn.Module):
    def __init__(self, ndim, r, hdl):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ndim, hdl),
            nn.PReLU(),
            nn.Linear(hdl, r),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z


class TgtOffHome(nn.Module):
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hdl, nout),
        )

    def forward(self, z, y=None):
        out = self.decoder(z)
        return out


class AdvOffHome(nn.Module):
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.adv = nn.Sequential(
            nn.Linear(r, nout)
        )

    def forward(self, z):
        out = self.adv(z)
        return out
