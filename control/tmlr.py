# tmlr.py

import torch
from collections import OrderedDict

import hal.kernels as kernels
import hal.models as models
import hal.utils.misc as misc
from control.base import BaseClass
import control.build_kernel as build_kernel
from tqdm import tqdm
import numpy as np


__all__ = ['TMLR']

class TMLR(BaseClass):

    def __init__(self, opts, dataloader):
        super().__init__(opts, dataloader)

        self.rff_flag = self.hparams.rff_flag
        self.kernel_x       = getattr(kernels, self.hparams.kernel_x)(**self.hparams.kernel_x_options)
        self.kernel_y       = getattr(kernels, self.hparams.kernel_y)(**self.hparams.kernel_y_options)
        self.kernel_s       = getattr(kernels, self.hparams.kernel_s)(**self.hparams.kernel_s_options)


        self.dataloader = dataloader

        self.model_device = 'cuda' if self.hparams.ngpu else 'cpu'
        
        # Initializing the kernel
        print('Initializing the kernel ...')

        self.compute_kernel(init=True)
        
        print('Initializing the kernel done!')


    def compute_kernel(self, features=None, Y=None, S=None, init=False):
        self.encoder = None
        if init:
            print('Initializing the kernel ...')
        else:
            print('Computing the kernel ...')

        features, Y, S = self.dataloader.train_kernel_dataloader()

        self.encoder = getattr(build_kernel, self.hparams.build_kernel)(self, features, Y, S)

        if init:
            print('Initializing the kernel is done!')
        else:
            print('Computing the kernel is done!')



    def training_step(self, batch, batch_idx, *args, **kwargs):     
        x, y, s = batch
        
        opt = self.optimizers()
        with torch.no_grad():
            z = self.encoder(x) # Kernel

        y_hat = self.target(z)

        loss_tgt = self.criterion['target'](y_hat, y)

        opt.zero_grad()
        self.manual_backward(loss_tgt)
        opt.step()
        self.used_optimizers[0] = True

        output = OrderedDict({
            'loss': loss_tgt.detach(),
            'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
            'y_hat': y_hat.detach(),
            'x': x,
            'z': z.detach(),
            'y': y.detach(),
            's': s.detach(),
        })
        return output


    def validation_step(self, batch, _):
        x, y, s = batch

        z = self.encoder(x)

        y_hat = self.target(z)

        loss_tgt = self.criterion['target'](y_hat, y)

        output = OrderedDict({
            'loss': loss_tgt.detach(),
            'x': x,
            's': s,
            'y_hat': y_hat,
            'y': y,
            'z': z.detach(),
            'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
        })
        return output

    def test_step(self, batch, _):
        x, y, s = batch

        z = self.encoder(x)

        y_hat = self.target(z)

        loss_tgt = self.criterion['target'](y_hat, y)

        output = OrderedDict({
            'loss': loss_tgt.detach(),
            'x': x,
            's': s,
            'y_hat': y_hat,
            'y': y,
            'z': z.detach(),
            'loss_tgt': {'value': loss_tgt.detach(), 'numel': len(x)},
        })
        return output

    def format_y_onehot(self, y):
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        y_onehot = torch.zeros(y.size(0), self.hparams.model_options['target']['nout'], device=y.device).scatter_(1, y.type(torch.int64), 1)
        return y_onehot
        
        
    def format_s_onehot(self, s):
        # int -> one-hot
        if len(s.shape) > 2:
            s = s.squeeze(-1)
        elif len(s.shape) == 1:
            s = s.unsqueeze(1)
            
        s_onehot = torch.zeros(s.size(0), self.hparams.metric_control_options['SP']['num_s_classes'], device=s.device).scatter_(1, s.long(), 1)
        return s_onehot
