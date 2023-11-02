# celeba.py

from torch import nn

__all__ = ['EncDecCelebAB']

class EncDecCelebAB(nn.Module):
    def __init__(self, ndim, nclasses, r, hdl):
        super(EncDecCelebA, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ndim, hdl),
            nn.PReLU(),
            nn.BatchNorm1d(hdl),
            nn.Linear(hdl, int(hdl / 2)),
            nn.PReLU(),
            nn.BatchNorm1d(int(hdl / 2)),
            nn.Linear(int(hdl / 2), r))

        self.decoder = nn.Sequential(
            nn.Linear(r, int(hdl / 2)),
            nn.PReLU(),
            # nn.Linear(int(hdl / 2), hdl),
            # nn.PReLU(),
            nn.Linear(int(hdl / 2), nclasses),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return z, out