# equalized_odds.py

import torch
import torchmetrics.metric as tm
from typing import Any, Callable, Optional

from hal.utils.misc import DetachableDict

__all__ = ['EqualizedOdds']


class EqualizedOdds(tm.Metric):
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,                 
                 n_attributes_c: int = 1,
                 n_classes_c: list = [2]):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        
        if len(n_classes_c) != n_attributes_c:
            n_classes_c = n_classes_c * n_attributes_c
        
        for cntl in range(n_attributes_c):
            self.add_state("total_" + str(cntl), default=torch.zeros((n_classes_c[cntl])), dist_reduce_fx=None)
            self.add_state("count_" + str(cntl), default=torch.zeros((n_classes_c[cntl])), dist_reduce_fx=None)

        self.n_classes_c = n_classes_c
        self.n_attributes_c = n_attributes_c

    def update(self, preds, target, control):
        if isinstance(preds, list):
            preds = preds.squeeze()
        if isinstance(target, list):
            target = target.squeeze()
        if isinstance(control, list):
            control = control.squeeze()

        assert preds.shape[1] == 2
        assert control.shape[1] == self.n_attributes_c
        pred = preds.max(1)[1]

        for cntl in range(self.n_attributes_c):
            for c_temp in range(self.n_classes_c):
                getattr(self, "total_" + str(cntl))[cntl, c_temp] += sum((control[:, cntl] == c_temp) & (target == 1))
                getattr(self, "count_" + str(cntl))[cntl, c_temp] += sum((pred == 1) & (control[:, cntl] == c_temp) & (target == 1))

    def compute(self):
        output = DetachableDict()
        for cntl in range(self.n_attributes_c):
            diff = []
            prob = getattr(self, "count_" + str(cntl)) / torch.clamp(getattr(self, "total_" + str(cntl)), 1.0)
            # calculate the difference between all pairs of probabilities
            for c1 in range(self.n_classes_c):
                for c2 in range(c1 + 1, self.n_classes_c):
                    diff.append(abs(prob[cntl, c1] - prob[cntl, c2]))
            
            output['EO_c_' + str(cntl)] = torch.Tensor(diff).max()

        return output