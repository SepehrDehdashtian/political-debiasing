# demographic_auroc.py

import torch
import torchmetrics.metric as tm
from typing import Any, Callable, Optional
import hal.metrics as metrics
from hal.utils.misc import DetachableDict

_all_ = ['DemographicAUROC']


class DemographicAUROC(tm.Metric):
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        
        self.add_state("control", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")

    def update(self, preds, target, control):
        '''
        preds :: (batch_size, n_classes_y) : Prediction of the model
        target :: (batch_size) : Target Labels
        control :: (batch_size, n_attributes_c) : Control labels across attributes
        '''
        if isinstance(preds, list):
            preds = preds.squeeze()
        if isinstance(control, list):
            control = control.squeeze()
        if isinstance(target, list):
            target = target.squeeze()

        self.target.append(target)
        self.control.append(control)
        self.preds.append(preds)

    def compute(self):
        self.target = torch.Tensor(self.target)
        self.control = torch.Tensor(self.control)
        self.preds = torch.Tensor(self.preds)

        output = DetachableDict()
        for cntl in range(self.control.shape[1]):
            c_values = len(self.control[:, cntl].unique())
            for i in range(c_values):
                preds = self.preds[self.control[:, cntl] == i]
                target = self.target[self.control[:, cntl] == i]
                output['control_' + str(cntl) + '_c_' + str(i)] = metrics.functional.auroc(preds, target)

        return output