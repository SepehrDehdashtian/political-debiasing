# general.py

import torch.nn as nn
import torch

__all__ = ['ResMLP', 'MLP', 'Adversary', 'HGRAdversary', 'Identity']


class MLP(nn.Module):
    def __init__(self, n_layers, input_size,
                 hidden_size, output_size,
                 dropout=0.2,
                 use_bn=1,
                 activation_fn=nn.PReLU):
        """
        Implements general MLP.
        Args:
            n_layers: number of hidden layers in MLP
            input_size: dimensionality of input features
            hidden_size: dimensionality of hidden features
            output_size: dimensionality of output features
            activation_fn: the activation function to use.
        """
        super(MLP, self).__init__()
        use_bn = bool(use_bn)

        model_list = []
        
        model_list.append(nn.Linear(input_size, hidden_size))
        if n_layers > 1:
            model_list.append(activation_fn())
            if use_bn:
                model_list.append(nn.BatchNorm1d(hidden_size))
            for i in range(n_layers-2):
                model_list.append(nn.Linear(hidden_size, hidden_size))
                model_list.append(activation_fn())
                if dropout != 0:
                    model_list.append(nn.Dropout(dropout))
                if use_bn:
                    model_list.append(nn.BatchNorm1d(hidden_size))
            model_list.append(nn.Linear(hidden_size, output_size, bias=False))
        
        self.model = nn.Sequential(*model_list)
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 0.25 is the initial slope of PReLU
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.25))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, y=None):
        x = self.model(x)
        ###### Normalization #####
        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-16)
        return x

class ResMLP(nn.Module):
    def __init__(self, n_layers, input_size,
                 hidden_size, output_size,
                 dropout=0.2, use_bn=1,
                 activation_fn=nn.GELU):
        """
        Implements general MLP with residual connections.
        Args:
            n_layers: number of hidden layers in MLP
            input_size: dimensionality of input features
            hidden_size: dimensionality of hidden features
            output_size: dimensionality of output features
            activation_fn: the activation function to use.
        """
        super().__init__()
        use_bn = bool(use_bn)
        if use_bn:
            self.first_block = nn.Sequential(nn.Linear(input_size, hidden_size),
                                            activation_fn(),
                                            nn.BatchNorm1d(hidden_size))
        else:
            self.first_block = nn.Sequential(nn.Linear(input_size, hidden_size),
                                            activation_fn())
        self.intermediate_blocks = nn.ModuleList()
        if n_layers > 1:
            for i in range(n_layers-1):
                temp = []
                temp.append(nn.Linear(hidden_size, hidden_size))
                temp.append(activation_fn())
                if dropout > 0:
                    temp.append(nn.Dropout(dropout))
                if use_bn:
                    temp.append(nn.BatchNorm1d(hidden_size))        
                self.intermediate_blocks.append(nn.Sequential(*temp))
        
        self.last_layer = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x, y=None):
        x = self.first_block(x)

        if len(self.intermediate_blocks) > 0:
            orig = x
            for m in self.intermediate_blocks:
                x = orig + m(x)
                orig = x

        x = self.last_layer(x)
        return x


class Adversary(nn.Module):
    def __init__(self, r=2, hdl=64, nclasses=4):
        super(Adversary, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(r, hdl),
            nn.PReLU(),
            nn.Linear(hdl, int(hdl / 2)),
            nn.PReLU(),
            nn.Linear(int(hdl / 2), r),
            nn.PReLU(),
            nn.Linear(r, nclasses),
        )

    def forward(self, x, y=None):
        out = self.decoder(x)
        return out

class HGRAdversary(nn.Module):
    def __init__(self, adversary_s, adversary_z):
        super().__init__()

        self.adversary_z = adversary_z
        self.adversary_s = adversary_s

    def forward(self, z, s):
        out_z = self.adversary_z(z)
        out_s = self.adversary_s(s)
        return out_z, out_s

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, *args):
        return z
