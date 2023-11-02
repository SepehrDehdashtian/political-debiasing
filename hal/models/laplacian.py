# laplacian.py

import torch
from torch import nn

__all__ = ['TraceLaplacian', 'NormLaplacian']

class TraceLaplacian(nn.Module):
    def __init__(self, L):
        super(TraceLaplacian, self).__init__()

        self.L = L.cpu()

    def forward(self, z):
        z = z.cpu()
        out = torch.mm(torch.mm(z.t(), self.L), z)
        
        out = torch.trace(out) / len(z)

        return out.cuda()


class NormLaplacian(nn.Module):
    def __init__(self, L):
        super(NormLaplacian, self).__init__()

        self.L = L.cpu()

    def forward(self, z):
        z = z.cpu()
        out = torch.mm(torch.mm(z.t(), self.L), z)
        
        out = torch.norm(out) / len(z)

        return out.cuda()
