# inverse_multi_quadratic.py

import torch

__all__ = ['InverseMultiquadric']

class InverseMultiquadric:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, x, s=None):
        sigma = self.sigma
        if s is None:
            dist = torch.cdist(x, x, p=2) ** 2
            kernel = sigma / torch.sqrt(dist + sigma ** 2)
        else:
            dist = torch.cdist(x, s, p=2) ** 2
            kernel = sigma / torch.sqrt(dist + sigma ** 2)

        return kernel
