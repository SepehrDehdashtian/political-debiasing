# continuous_fairness.py

import torch
from torch import nn
from math import pi, sqrt

__all__ = ['ChiSquaredLoss']

"""
Code taken from:
https://github.com/criteo-research/continuous-fairness
"""

def _unsqueeze_multiple_times(input, axis, times):
    """
    Utils function to unsqueeze tensor to avoid cumbersome code
    :param input: A pytorch Tensor of dimensions (D_1,..., D_k)
    :param axis: the axis to unsqueeze repeatedly
    :param times: the number of repetitions of the unsqueeze
    :return: the unsqueezed tensor. ex: dimensions (D_1,... D_i, 0,0,0, D_{i+1}, ... D_k) for unsqueezing 3x axis i.
    """
    output = input
    for i in range(times):
        output = output.unsqueeze(axis)
    return output

class kde:
    """
    A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization.
    Keep in mind that KDE are not scaling well with the number of dimensions and this implementation is not really
    optimized...
    """
    def __init__(self, x_train):
        n, d = x_train.shape

        self.n = n
        self.d = d

        self.bandwidth = (n * (d + 2) / 4.) ** (-1. / (d + 4))
        self.std = self.bandwidth

        self.train_x = x_train

    def pdf(self, x):
        s = x.shape
        d = s[-1]
        s = s[:-1]
        assert d == self.d

        data = x.unsqueeze(-2)

        train_x = _unsqueeze_multiple_times(self.train_x, 0, len(s))

        pdf_values = torch.exp(-((data-train_x).norm(dim=-1)**2 / \
                               (self.bandwidth**2)/2)).mean(dim=-1) / sqrt(2*pi) / self.bandwidth

        return pdf_values

# Independence of 2 variables
def _joint_2(X, Y, damping=1e-10):
    X = (X - X.mean(dim=0, keepdim=True)) / X.std(dim=0, keepdim=True)
    Y = (Y - Y.mean(dim=0, keepdim=True)) / Y.std(dim=0, keepdim=True)
    data = torch.cat([X.unsqueeze(-1), Y.unsqueeze(-1)], 1) # estimation of joint probability?
    joint_density = kde(data)

    nbins = int(min(50, 5. / joint_density.std))
    #nbins = np.sqrt( Y.size/5 )
    x_centers = torch.linspace(-2.5, 2.5, nbins)
    y_centers = torch.linspace(-2.5, 2.5, nbins)

    xx, yy = torch.meshgrid([x_centers, y_centers])
    grid = torch.cat([xx.unsqueeze(-1), yy.unsqueeze(-1)], -1)
    h2d = joint_density.pdf(grid) + damping
    h2d /= h2d.sum()
    return h2d

class ChiSquaredLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def chi_2(self, X, Y, damping = 0):
        """
        The \chi^2 divergence between the joint distribution on (x,y) and the product of marginals. This is know to be the
        square of an upper-bound on the Hirschfeld-Gebelein-Renyi maximum correlation coefficient. We compute it here on
        an empirical and discretized density estimated from the input data.
        :param X: A torch 1-D Tensor
        :param Y: A torch 1-D Tensor
        :param density: so far only kde is supported
        :return: numerical value between 0 and \infty (0: independent)
        """
        h2d = _joint_2(X, Y, damping=damping)
        marginal_x = h2d.sum(dim=1).unsqueeze(1)
        marginal_y = h2d.sum(dim=0).unsqueeze(0)
        Q = h2d / (torch.sqrt(marginal_x) * torch.sqrt(marginal_y))
        return ((Q ** 2).sum(dim=[0, 1]) - 1.)

    def __call__(self, inputs, targets):
        loss = self.chi_2(inputs, targets)
        return loss
