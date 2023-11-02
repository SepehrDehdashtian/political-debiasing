# arc_cosine.py

import torch

__all__ = ['ArcCosine', 'RFFArcCosine']

class ArcCosine:

    def __init__(self, order=None, rff=False):
        self.rff = rff
        self.order = order

    def __call__(self, x, y=None):
        kernel = 0
        return kernel


class RFFArcCosine:
    
    def __init__(self, order=None, rff=False):
        self.rff = rff
        self.order = order

    def __call__(self, x, y=None):
        kernel = 0
        return kernel