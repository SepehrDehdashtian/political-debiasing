# statistical_parity.py

import torch
import torchmetrics.metric as tm
from typing import Any, Callable, Optional
from hal.utils.misc import DetachableDict
import numpy as np

__all__ = ['StatisticalParity']


class StatisticalParity(tm.Metric):
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 n_classes_y: int = 2,
                 n_classes_c: list = [2],
                 n_attributes_c: int = 1):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if len(n_classes_c) != n_attributes_c:
            n_classes_c = n_classes_c * n_attributes_c

        for cntl in range(n_attributes_c):
            self.add_state("total_" + str(cntl), default=torch.zeros((n_classes_y, n_classes_c[cntl], n_classes_y)), dist_reduce_fx=None)
            self.add_state("count_" + str(cntl), default=torch.zeros((n_classes_y, n_classes_c[cntl], n_classes_y)), dist_reduce_fx=None)


        self.n_classes_y    = n_classes_y
        self.n_classes_c    = n_classes_c
        self.n_attributes_c = n_attributes_c

    def update(self, preds, control):
        if isinstance(preds, list):
            preds = preds.squeeze()
        if isinstance(control, list):
            control = control.squeeze()

        if len(control.shape) < 2: control = control.unsqueeze(1)
        
        assert control.shape[1] == self.n_attributes_c
        pred = preds.max(1)[1]

        
        for cntl in range(self.n_attributes_c):
                for c_temp in range(self.n_classes_c[cntl]):
                    for y_temp_hat in range(self.n_classes_y):
                        getattr(self, "total_" + str(cntl))[y_temp_hat, c_temp] += sum((control[:, cntl] == c_temp))
                        getattr(self, "count_" + str(cntl))[y_temp_hat, c_temp] += sum((pred == y_temp_hat) & (control[:, cntl] == c_temp))

    def compute(self):
        output = DetachableDict()
        for cntl in range(self.n_attributes_c):
            diff = torch.Tensor().cuda()
            diff2 = torch.Tensor().cuda()
            prob = getattr(self, "count_" + str(cntl)) / torch.clamp(getattr(self, "total_" + str(cntl)), 1.0)
            # calculate the difference between all pairs of probabilities
            for y_temp_hat in range(self.n_classes_y):
                    for c1 in range(self.n_classes_c[cntl]):
                        for c2 in range(c1 + 1, self.n_classes_c[cntl]):
                            diff  = torch.cat((diff, abs(prob[y_temp_hat, c1] - prob[y_temp_hat, c2])), 0)
                            diff2 = torch.cat((diff2, (prob[y_temp_hat, c1] - prob[y_temp_hat, c2])**2), 0)

            output['SP_max_' + str(cntl)] = diff.max()
            output['SP_mean_' + str(cntl)] = diff.mean()
            
            # output['SP_max_' + str(cntl)] = diff2.max()
            output['SP_sq_mean_' + str(cntl)] = np.sqrt(diff2.mean().cpu())
            
            # import pdb; pdb.set_trace()
            # output['SP_var_mean_' + str(cntl)] = 





        return output