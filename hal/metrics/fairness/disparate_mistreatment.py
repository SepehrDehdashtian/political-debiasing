# disparate_mistreatment.py

import torch
import torchmetrics.metric as tm
from typing import Any, Callable, Optional

from hal.utils.misc import DetachableDict

__all__ = ['DisparateMistreatment']


class DisparateMistreatment(tm.Metric):
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 n_attributes_c: int = 2):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("num_00", default=torch.zeros(n_attributes_c), dist_reduce_fx=None)
        self.add_state("num_01", default=torch.zeros(n_attributes_c), dist_reduce_fx=None)
        self.add_state("num_10", default=torch.zeros(n_attributes_c), dist_reduce_fx=None)
        self.add_state("num_11", default=torch.zeros(n_attributes_c), dist_reduce_fx=None)

        self.add_state("den_00", default=torch.zeros(n_attributes_c), dist_reduce_fx=None)
        self.add_state("den_01", default=torch.zeros(n_attributes_c), dist_reduce_fx=None)
        self.add_state("den_10", default=torch.zeros(n_attributes_c), dist_reduce_fx=None)
        self.add_state("den_11", default=torch.zeros(n_attributes_c), dist_reduce_fx=None)

    def update(self, preds, target, control):
        if isinstance(preds, list):
            preds = preds.squeeze()
        if isinstance(control, list):
            control = control.squeeze()
        if isinstance(target, list):
            target = target.squeeze()

        assert control.shape[1] == self.n_attributes_c
        pred = preds.max(1)[1]

        for cntl in range(self.n_attributes_c):
            self.num_00[cntl] += sum((pred == 1) & (control[:, cntl] == 0) & (target == 0))
            self.num_01[cntl] += sum((pred == 1) & (control[:, cntl] == 1) & (target == 0))
            self.num_10[cntl] += sum((pred == 0) & (control[:, cntl] == 0) & (target == 1))
            self.num_11[cntl] += sum((pred == 0) & (control[:, cntl] == 1) & (target == 1))

            self.den_00[cntl] += sum((control[:, cntl] == 0) & (target == 0))
            self.den_01[cntl] += sum((control[:, cntl] == 1) & (target == 0))
            self.den_10[cntl] += sum((control[:, cntl] == 0) & (target == 1))
            self.den_11[cntl] += sum((control[:, cntl] == 1) & (target == 1))

    def compute(self):
        output = DetachableDict()
        for cntl in range(self.n_attributes_c):
            dm_fpr = self.num_00[cntl] / self.den_00[cntl] - self.num_01[cntl] / self.den_01[cntl]
            dm_fnr = self.num_10[cntl] / self.den_10[cntl] - self.num_11[cntl] / self.den_11[cntl]
            output['FPR_c_' + str(cntl)] = dm_fpr
            output['FNR_c_' + str(cntl)] = dm_fnr
        return output