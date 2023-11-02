# spatial_transformer_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['STN']

class nnSqueeze(nn.Module):
    def __init__(self):
        super(nnSqueeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)

def convert_Avec_to_A(A_vec):
    """ Convert BxM tensor to BxNxN symmetric matrices """
    """ M = N*(N+1)/2"""
    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze(dim=0)

    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 3:
        A_dim = 2
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")

    idx = torch.triu_indices(A_dim, A_dim)
    A = A_vec.new_zeros((A_vec.shape[0], A_dim, A_dim))
    A[:, idx[0], idx[1]] = A_vec
    A[:, idx[1], idx[0]] = A_vec
    return A.squeeze()

class STN(nn.Module):

    def __init__(self, inp_dim, out_dim):
        super().__init__()

        nh = 32

        self.fc_stn = nn.Sequential(
            nn.Linear(inp_dim, nh), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(nh, nh), nn.BatchNorm1d(nh), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            nn.Linear(nh, 10),
        )

        nz = 10

        self.conv = nn.Sequential(
            nn.Conv2d(1, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(0.2),  # 14 x 14
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(0.2),  # 7 x 7
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True), nn.Dropout(0.2),  # 4 x 4
            nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True),  # 1 x 1
        )

        # self.reduce_dim = nn.Linear(1280, out_dim)

    def stn(self, x):
        A_vec = self.fc_stn(x)
        A = convert_Avec_to_A(A_vec)
        _, evs = torch.symeig(A, eigenvectors=True)
        tcos, tsin = evs[:, 0:1, 0:1], evs[:, 1:2, 0:1]

        self.theta_angle = torch.atan2(tsin[:, 0, 0], tcos[:, 0, 0])

        # clock-wise rotate theta
        theta_0 = torch.cat([tcos, tsin, tcos * 0], 2)
        theta_1 = torch.cat([-tsin, tcos, tcos * 0], 2)
        theta = torch.cat([theta_0, theta_1], 1)

        batchsize = x.size(0)
        img = torch.reshape(x[:, :-1], (batchsize, 1, 28, 28))
        grid = F.affine_grid(theta, img.size(), align_corners=False)
        x = F.grid_sample(img, grid, align_corners=False)
        return x

    def forward(self, x):
        """
        :param x: B x 1 x 28 x 28
        :param u: B x nu
        :return:
        """
        x = self.stn(x)
        z = self.conv(x)

        h, w = z.size(2), z.size(3)
        z = F.avg_pool2d(z, (h, w), stride=(h, w)).squeeze()

        return z