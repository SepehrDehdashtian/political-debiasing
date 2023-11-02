# german.py

from torch import nn

__all__ = ['EncDecGerman']


class EncDecGerman(nn.Module):

    def __init__(self, ndim, r, nclasses):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ndim, r, bias=True),
            nn.SELU(),
            nn.Linear(r, r, bias=False),
        )
        self.bn = nn.BatchNorm1d(r)
        self.decoder = nn.Sequential(nn.Linear(r, nclasses, bias=False), )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return z, out