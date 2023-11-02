# credit.py

from torch import nn
# from torch_geometric.nn import GCNConv
# from dgl.nn.pytorch import GATConv
# from dgl.nn.pytorch import GraphConv
# # from torch_geometric.nn import GINConv
# from dgl.nn import GINConv

__all__ = ['EncCreditGIN', 'EncCredit', 'EncCreditGCN', 'TgtCredit', 'AdvCredit']


class EncCredit(nn.Module):  # hdl:128 r:1
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
        
class EncCreditGIN(nn.Module):
    def __init__(self, ndim, hdl, r, dropout): 
        super(EncCreditGIN, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(ndim, hdl), 
            nn.ReLU(),
            nn.BatchNorm1d(hdl),
            nn.Linear(hdl, hdl), 
        )
        self.conv1 = GINConv(self.mlp1)
        self.fc = nn.Linear(hdl, r)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
        
    def forward(self, x, dglG): 
        z = self.conv1(dglG, x)
        z = self.fc(z)
        return z


class EncCreditGCN(nn.Module):
    def __init__(self, ndim, r, hdl=None):
        super().__init__()
        self.gc1 = GraphConv(ndim, hdl)
        self.gc2 = GraphConv(hdl, r)
        # self.gc3 = GraphConv(hdl//2, r)

    def forward(self, x, dglG, y=None):
        # import pdb; pdb.set_trace()
        z = self.gc1(dglG, x)
        z = self.gc2(dglG, z)
        # z = self.gc3(dglG, z)
        return z


class TgtCredit(nn.Module):  # hdl:128
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.dec = nn.Sequential(
            nn.Linear(r, nout)
        )

        self.dec_3Layer = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl // 2),
            nn.PReLU(),
            nn.Linear(hdl // 2, nout)
        )

        self.dec_2Layer = nn.Sequential(
            nn.Linear(r, hdl // 2),
            nn.PReLU(),
            nn.Linear(hdl // 2, nout)
        )

    def forward(self, z, y=None):
        # out = self.dec(z)
        # out = self.dec_3Layer(z)
        out = self.dec_2Layer(z)
        return out


class AdvCredit(nn.Module):
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