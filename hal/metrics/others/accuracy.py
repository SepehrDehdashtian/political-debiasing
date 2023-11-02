"""
UtilityFromMSE is a wrapper around MSE function. MSE function gives the
error and we want to convert it to a quantity of utility.

In regression tasks, the worst prediction by a model is the expected
value of the target. Utility of a model predicting y_hat is given by,

U = 1 - MSE(y_hat, y)/MSE(E[Y], y)

For now, this file is kept under fairness. Later on, this would be moved
to another location.
"""

import torch
import torchmetrics as tm
from typing import Any, Callable, Optional

__all__ = ['AccuracyUtility', 'AccuracyUtilityForDomInd', 'AccuracyUtilityMultiLabel']

class AccuracyUtilityMultiLabel(tm.Metric):
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("xx", default=[], dist_reduce_fx=None) # estimate
        self.add_state("yy", default=[], dist_reduce_fx=None) # GT

    def update(self, xx, yy):
        self.xx.append(xx)
        self.yy.append(yy)

    def compute(self):
        xx = torch.cat(self.xx, 0)
        y_hat = (xx > 0).float()
        yy = torch.cat(self.yy, 0)

        acc = torch.zeros(y_hat.shape[1])
        for k in range(y_hat.shape[1]):
            acc[k] = torch.mean((y_hat[:, k] == yy[:, k]).type(torch.float))

        return acc.mean()

class AccuracyUtility(tm.Metric):
    def __init__(self,
                 one_hot=0,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.one_hot = bool(one_hot)

        self.add_state("xx", default=[], dist_reduce_fx=None) # estimate
        self.add_state("yy", default=[], dist_reduce_fx=None) # GT

    def update(self, xx, yy):
        # import pdb; pdb.set_trace()
        self.xx.append(xx)
        self.yy.append(yy)

    def compute(self):
        # import pdb; pdb.set_trace()
        xx = torch.cat(self.xx, 0)
        xx = torch.argmax(xx, 1)
        yy = torch.cat(self.yy, 0)
        if self.one_hot:
            yy = torch.argmax(yy, 1)

        acc = torch.mean((xx==yy).type(torch.float))

        return acc

class AccuracyUtilityForDomInd(tm.Metric):
    def __init__(self,
                 num_classes,
                 one_hot=1,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.num_classes = num_classes
        self.one_hot = one_hot

        self.add_state("xx", default=[], dist_reduce_fx=None) # estimate
        self.add_state("yy", default=[], dist_reduce_fx=None) # GT

    def update(self, xx, yy):
        """
        xx is of shape (batch_size, num_domains*num_classes).
        Its dimension 1 is arranged as [dom1_cls1, dom1_cls2, ...,
        dom1_clsN, dom2_cls1, dom2_cls2, ..., dom2_clsN, ...,
        domM_clsN]. We will sum the logit values for each class across
        domains and then take the argmax.
        """
        num_domains = xx.size(1)//self.num_classes
        xx = torch.reshape(xx, (-1, num_domains, self.num_classes))
        xx = torch.sum(xx, dim=1)
        # Argmax happens in compute()
        self.xx.append(xx)
        self.yy.append(yy)

    def compute(self):
        xx = torch.cat(self.xx, 0)
        xx = torch.argmax(xx, 1)
        yy = torch.cat(self.yy, 0)
        if self.one_hot:
            yy = torch.argmax(yy, 1)

        acc = torch.mean((xx==yy).type(torch.float))

        return acc
