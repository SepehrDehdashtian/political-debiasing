# HateSpeech.py

import torch
from torch import nn

# from torch_geometric.nn import GCNConv
# from dgl.nn.pytorch import GATConv
# from dgl.nn.pytorch import GraphConv
# # from torch_geometric.nn import GINConv
# from dgl.nn import GINConv

__all__ = ['EncHateSpeech', 'TgtHateSpeech', 'SmallTgtHateSpeech', 'SmallTgtHateSpeech2', 'AdvHateSpeech', 'HateSpeechEmbeddingYS']


# class EncHateSpeech(nn.Module):  # hdl:128 r:1
#     def __init__(self, ndim, r, hdl):
#         super().__init__()

#         self.enc_equal = nn.Sequential(
#             nn.Linear(ndim, hdl),
#             nn.PReLU(),
#             nn.Linear(hdl, hdl // 2),
#             nn.PReLU(),
#             nn.Linear(hdl // 2, r)
#         )

#     def forward(self, x, y=None):
#         z = self.enc_equal(x)
#         return z

class EncHateSpeech(nn.Module):  # hdl:128 r:1
    def __init__(self, ndim, r, hdl):
        super().__init__()

        self.enc_equal = nn.Sequential(
            # nn.Linear(ndim, hdl),
            # nn.PReLU(),
            # nn.Linear(hdl, hdl // 2),
            # nn.PReLU(),
            nn.Linear(ndim, r)
        )

    def forward(self, x, y=None):
        z = self.enc_equal(x)
        return z


class TgtHateSpeech(nn.Module):  # hdl:128
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.dec_equal2 = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl // 2),
            nn.PReLU(),
            nn.Linear(hdl // 2, nout)
        )

    def forward(self, z, y=None):
        out = self.dec_equal2(z)
        return out

class SmallTgtHateSpeech(nn.Module):  # hdl:128
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.dec_2Layer = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, nout)
        )

    def forward(self, z, y=None):
        out = self.dec_2Layer(z)
        return out

class SmallTgtHateSpeech2(nn.Module):  # hdl:128
    def __init__(self, nout, r, hdl):
        super().__init__()

        self.dec_equal2 = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, hdl // 2),
            nn.PReLU(),
            nn.Linear(hdl // 2, nout)
        )

    def forward(self, z, y=None):
        out = self.dec_equal2(z)
        return out


class AdvHateSpeech(nn.Module):
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
    

class HateSpeechEmbeddingYS(nn.Module):
    def __init__(self):
        super(HateSpeechEmbeddingYS, self).__init__()
        
        self.embedd_y = nn.Embedding(2, 1)
        self.embedd_s = nn.Embedding(4, 2)

        # self.fc1 = nn.Linear(sum(num_embedding) + 1, 64)

    def forward(self, x):
        # Apply the embeddings to the features and concat. them
        x0 = self.embedd_s(x[:,0])
        x1 = self.embedd_y(x[:,1])

        # Add age attribute to the embeddings
        embedded = torch.cat((x0, x1), dim=1).float()

        # out = self.fc1(embedded)
        
        return embedded   