import torch
import torch.nn as nn

class ReturnZero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, s, *args):
        return torch.tensor(0.0).to(z.device)