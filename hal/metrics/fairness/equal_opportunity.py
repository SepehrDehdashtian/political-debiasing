# equal_opportunity.py

import torch
import torchmetrics.metric as tm
from typing import Any, Callable, Optional
# from torchmetrics.utilities import check_forward_no_full_state
from hal.utils.misc import DetachableDict

__all__ = ['EqualOpportunity']


class EqualOpportunity(tm.Metric):
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 # full_state_update=False,
                 n_classes_y: int = 2,
                 n_classes_c: list = [2],
                 n_attributes_c: int = 1):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            # full_state_update=False,
        )

        if len(n_classes_c) != n_attributes_c:
            n_classes_c = n_classes_c * n_attributes_c

        for cntl in range(n_attributes_c):
            self.add_state("total_" + str(cntl), default=torch.zeros((n_classes_y, n_classes_c[cntl], n_classes_y)), dist_reduce_fx=None)
            self.add_state("count_" + str(cntl), default=torch.zeros((n_classes_y, n_classes_c[cntl], n_classes_y)), dist_reduce_fx=None)

        # import pdb; pdb.set_trace()

        self.n_classes_y    = n_classes_y
        self.n_classes_c    = n_classes_c
        self.n_attributes_c = n_attributes_c

    def update(self, preds, target, control):
        if isinstance(preds, list):
            preds = preds.squeeze()
        if isinstance(control, list):
            control = control.squeeze()
        if isinstance(target, list):
            target = target.squeeze()

        if len(control.shape) < 2: control = control.unsqueeze(1)
        
        assert control.shape[1] == self.n_attributes_c
        pred = preds.max(1)[1]

        for cntl in range(self.n_attributes_c):
            for y_temp in range(self.n_classes_y):
                for c_temp in range(self.n_classes_c[cntl]):
                    for y_temp_hat in range(self.n_classes_y):
                        getattr(self, "total_" + str(cntl))[y_temp_hat, y_temp, c_temp] += sum((control[:, cntl] == c_temp) & (target == y_temp))
                        getattr(self, "count_" + str(cntl))[y_temp_hat, y_temp, c_temp] += sum((pred == y_temp_hat) & (control[:, cntl] == c_temp) & (target == y_temp))

    def compute(self):
        output = DetachableDict()
        for cntl in range(self.n_attributes_c):
            diff = []
            prob = getattr(self, "count_" + str(cntl)) / torch.clamp(getattr(self, "total_" + str(cntl)), 1.0)
            # calculate the difference between all pairs of probabilities
            for y_temp_hat in range(self.n_classes_y):
                for y_temp in range(self.n_classes_y):
                    for c1 in range(self.n_classes_c[cntl]):
                        for c2 in range(c1 + 1, self.n_classes_c[cntl]):
                            diff.append(abs(prob[y_temp_hat, y_temp, c1] - prob[y_temp_hat, y_temp, c2]))
        
            output['EOO_c_' + str(cntl)] = torch.Tensor(diff).max()

        return output