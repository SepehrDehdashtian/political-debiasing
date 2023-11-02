# pokec.py
import torch
from torch import nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv

__all__ = ['EncLinear', 'EncPokec', 'TgtPokec', 'AdvPokec', 'RepPokecHGR', 'AdvPokecHGR', 'EncGAT', 'EncGCN']


class EncGAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):

        super(EncGAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.relu

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))

        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_hidden, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)

        return logits


class EncGCN(nn.Module):
    def __init__(self, ndim, r, dropout):
        super(EncGCN, self).__init__()

        self.gc1 = GraphConv(ndim, r, norm='both')
        self.gc2 = GraphConv(r, r, norm='both')
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(r,r)
        # self.fc2 = nn.Linear(r,r)
        self.fc3 = nn.Linear(r,r)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = self.dropout1(x)
        x = F.relu(self.gc2(g, x))
        # x = self.dropout2(x)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        z = self.fc3(x)
        
        return z


class EncLinear(nn.Module):

    def __init__(self, nout, r):
        super().__init__()

        # ndim: in features
        # r: out features

        self.linear = nn.Linear(r, nout, bias=True)

    def forward(self, x, y=None):
        z = self.linear(x)
        return z

class EncPokec(nn.Module):
    def __init__(self, ndim, r, hdl):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(ndim, 2*hdl),
            nn.PReLU(),
            nn.Linear(2*hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl//2),
            nn.PReLU(),
            nn.Linear(hdl//2, r, bias=False)
        )

    def forward(self, x, y=None):
        z = self.encoder(x)
        ###### Normalization #####
        z = z / (torch.norm(z, dim=1, keepdim=True) + 1e-16)
        return z


class TgtPokec(nn.Module):
    def __init__(self, nout, r, hdl):
        super().__init__()
        self.arl_target = nn.Sequential(
            nn.Linear(r, 2*hdl),
            nn.PReLU(),
            nn.Linear(2*hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

        self.GCN_larger_target = nn.Sequential(
            nn.Linear(r, 2*hdl),
            nn.PReLU(),
            nn.Linear(2*hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl//2),
            nn.PReLU(),
            nn.Linear(hdl//2, nout),
        )

        self.smaller_target = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

        self.smaller_target2 = nn.Sequential(
            nn.Linear(r, hdl//2),
            nn.PReLU(),
            nn.Linear(hdl//2, nout),
        )

    def forward(self, z, y=None):
        # import pdb; pdb.set_trace()
        out = self.arl_target(z)
        # out = self.smaller_target(z)
        # out = self.smaller_target2(z)
        # out = self.GCN_larger_target(z)
        return out

class AdvPokec(nn.Module):

    def __init__(self, nout, r, hdl):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(r, 2*hdl),
            nn.PReLU(),
            nn.Linear(2*hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

    def forward(self, x, y=None):
        out = self.decoder(x)
        return out

class RepPokecHGR(nn.Module):

    def __init__(self, nout, r, hdl):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

    def forward(self, z, y=None):
        out = self.target(z)
        return out


class AdvPokecHGR(nn.Module):

    def __init__(self, nout, ndim, hdl):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(ndim, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout),
        )

    def forward(self, x, y=None):
        out = self.decoder(x)
        return out



