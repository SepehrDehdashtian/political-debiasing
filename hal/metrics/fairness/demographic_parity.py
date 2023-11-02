# demographic_parity.py

import torch
import numpy as np
import torchmetrics.metric as tm
from typing import Any, Callable, Optional

import torch.nn.functional as F

from hal.utils.misc import DetachableDict

__all__ = ['DemographicParity', 'DemographicParityQuantize', 'DemographicParityStrong', 'DemographicParityUtilityMSE',
]

def discretize_(x: torch.Tensor, levels: list) -> torch.Tensor:
    """
    Discretize a 1-D tensor according to the levels.
    """
    out = x.clone()
    out = out.reshape(-1, 1)
    out = torch.tile(out, (1, len(levels)))
    levels = torch.tensor(levels, dtype=torch.float32, device=out.device).reshape(1, -1)
    levels = torch.tile(levels, (out.shape[0], 1))
    cost = torch.abs(out - levels)
    out = torch.argmin(cost, dim=1).reshape(-1)

    return out

def discretize(x: torch.Tensor, levels: list, sens=False) -> torch.Tensor:
    """
    Discretize a multi-dimensional tensor according to the levels.
    """
    out_list = []
    for i in range(x.shape[1]):
        out_ = discretize_(x[:, i], levels).reshape(-1, 1)
        out_list.append(out_)
    out = torch.cat(out_list, dim=1).float()

    if not sens:
        # Combine the dimensions using multiplexing
        # Right now, out is of shape (batch_size, num_dims)
        with torch.no_grad():
            powers = torch.arange(out.size(1)).float()
            base = float(len(levels))
            out = torch.mm(out, torch.pow(base, powers).to(out.device).reshape(-1, 1))

        out = out.reshape(-1)

    return out

def quantize(x: torch.Tensor, num: int) -> torch.Tensor:
    """
    Quantize a probability to num quantiles.
    """
    levels = torch.linspace(0, 1, steps=num).to(device=x.device)
    indeces = (levels.repeat(*x.shape, 1) - x.unsqueeze(-1)).abs().argmin(dim=-1)
    # for k in range(x.shape[0]):
    #     out[k, :] = levels[indeces[k, :]]
    # out = levels[indeces]

    with torch.no_grad():
        powers = torch.arange(indeces.size(1)).float()
        base = float(len(levels))
        out_ind = torch.mm(indeces.float(), torch.pow(base, powers).to(x.device).reshape(-1, 1))

    out_ind = out_ind.reshape(-1)

    return out_ind

def quantize_eq1(x: torch.Tensor, num: int) -> torch.Tensor:
    """
    Quantize a probability to num quantiles.
    """
    n_eq = int(x.shape[0] / (num))

    x_sorted, _ = torch.sort(x, dim=0)
    levels = torch.zeros((num+1,), device=x.device)
    indeces = torch.zeros_like(x)

    for k in range(num-1):
        levels[k+1] = x_sorted[(k+1)*n_eq]
        indeces[(x > levels[k]) & (x <= levels[k+1])] = k
    levels[-1] = 1.0
    indeces[(x > levels[num-1]) & (x <= levels[num])] = num-1

    # with torch.no_grad():
    #     powers = torch.arange(indeces.size(1)).float()
    #     base = float(len(levels)-1)
    #     out_ind = torch.mm(indeces.float(), torch.pow(base, powers).to(x.device).reshape(-1, 1))
    indeces = indeces.reshape(-1)

    return indeces

def threshold_minmax(x :torch.Tensor, min_num: int):
    x_sorted, _ = torch.sort(x, dim=0)
    threshold_min = x_sorted[min_num]
    threshold_max = x_sorted[-min_num]

    return threshold_min, threshold_max

def quantize_eq2(x: torch.Tensor, num: int) -> torch.Tensor:
    """
    Quantize a probability to num quantiles.
    """
    n_eq = int(x.shape[0] / (num))

    x_sorted, x_indices = torch.sort(x[:, 0], dim=0)
    levels = torch.zeros((num+1, num+1), device=x.device)
    levels0 = torch.zeros((num+1,), device=x.device)
    indeces = torch.zeros_like(x[:, 0])

    for k in range(num-1):
        levels0[k+1] = x_sorted[(k+1)*n_eq]
        # indeces[(x[:, 0] > levels[k, 0]) & (x[:, 0] <= levels[k+1, 0])] = k
    levels0[-1] = 1.0
    # indeces[(x[:, 0] > levels[num-1, 0]) & (x[:, 0] <= levels[num, 0])] = num-1

    for i in range(num):
        x_new = x[(x[:, 0] > levels0[i]) & (x[:, 0] <= levels0[i+1])]
        n_eq = int(x_new.shape[0] / (num))

        x_sorted, _ = torch.sort(x_new[:, 1], dim=0)

        for k in range(num - 1):

            levels[k + 1, i+1] = x_sorted[(k + 1) * n_eq]
            indeces[(x[:, 0] > levels0[i]) & (x[:, 0] <= levels0[i + 1]) & (x[:, 1] > levels[k, i+1]) & (x[:, 1] <= levels[k + 1, i+1])] = i*num+k
        levels[-1, i+1] = 1.0
        indeces[(x[:, 0] > levels0[i]) & (x[:, 0] <= levels0[i+1]) & (x[:, 1] > levels[num - 1, i+1]) & (x[:, 1] <= levels[num, i+1])] = i*num + num-1

    return indeces

class DemographicParityUtilityMSE(tm.Metric):

    def __init__(self,
                 mean,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 n_classes_s: int = 2,
                 discrete_s = None,
                 domind: bool = False):

        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)

        self.add_state("prediction", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)
        self.add_state("control", default=[], dist_reduce_fx=None)

        if isinstance(mean, list):
            mean = torch.Tensor(mean).reshape(1, -1)
        self.mean = mean

        self.n_classes_c = n_classes_s
        self.discrete_s = discrete_s

        if isinstance(domind, str):
            self.domind = True if domind.lower() == "true" else False
        else:
            self.domind = domind

    def update(self, preds, target, control):
        if self.discrete_s is not None:
            control = discretize(control, self.discrete_s, sens=False)
        else:
            if control.size(1) == self.n_classes_c[0]:
                assert len(self.n_classes_c) == 1
                control = control.argmax(dim=1)
        self.prediction.append(preds)
        self.target.append(target)
        self.control.append(control)

    def compute(self):

        preds = torch.cat(self.prediction, 0)
        target = torch.cat(self.target, 0)
        control = torch.cat(self.control, 0)

        total = torch.zeros(self.n_classes_c, device=preds.device)
        utility = torch.zeros(self.n_classes_c, device=preds.device)

        mean_vector = torch.tile(self.mean, (preds.shape[0], 1))
        mean_vector = mean_vector.to(preds.device)
        mean_error = F.mse_loss(mean_vector, target)

        for c_temp in range(self.n_classes_c):
            temp = (control == c_temp)
            total[c_temp] += sum(temp)
            utility0 = 1. - F.mse_loss(preds[temp], target[temp]) / mean_error
            utility[c_temp] = torch.maximum(torch.zeros_like(utility0), utility0)

        # ss_diff = torch.zeros((self.n_classes_c, self.n_classes_c), device=preds.device)
        # for c1 in range(self.n_classes_c):
        #     for c2 in range(c1 + 1, self.n_classes_c):
        #         sq_diff = torch.abs(utility[c1] - utility[c2])
        #         ss_diff[c1, c2] += sq_diff

        sq_s = torch.tile(utility, (len(utility), 1))
        ss_diff = torch.abs(sq_s - sq_s.transpose(1, 0))

        # import pdb; pdb.set_trace()

        prob_s = total / total.sum()
        prob_s = prob_s.unsqueeze(0)
        dp_avg = torch.mm(torch.mm(prob_s, ss_diff), prob_s.t()) # ss_diff is not PD rather upper triangular
        dp_max = ss_diff.max()

        output = DetachableDict()

        output["DP_avg"] = dp_avg
        output["DP_max"] = dp_max

        return output

class DemographicParityStrong(tm.Metric):

    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 min_num: int = 1000,
                 tau_step = 20,
                 n_classes_s: int = 2,
                 discrete_s = None,
                 domind: bool = False):

        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)


        self.add_state("prediction", default=[], dist_reduce_fx=None)
        self.add_state("control", default=[], dist_reduce_fx=None)

        self.min_num = min_num
        self.tau_step = tau_step
        self.n_classes_c = n_classes_s
        self.discrete_s = discrete_s

        if isinstance(domind, str):
            self.domind = True if domind.lower() == "true" else False
        else:
            self.domind = domind

    def update(self, preds, control):
        if self.discrete_s is not None:
            control = discretize(control, self.discrete_s, sens=False)
        else:
            if control.size(1) == self.n_classes_c[0]:
                assert len(self.n_classes_c) == 1
                control = control.argmax(dim=1)
        self.prediction.append(preds)
        self.control.append(control)

    def compute(self):

        preds = torch.cat(self.prediction, 0)
        control = torch.cat(self.control, 0)

        preds_n = (preds - torch.min(preds, dim=0)[0]) / (torch.max(preds, dim=0)[0] - torch.min(preds, dim=0)[0])

        output = DetachableDict()

        thre_min, thre_max = threshold_minmax(preds_n, self.min_num)

        if preds_n.shape[1] == 1:
            # tau_grid = torch.linspace(thre_min[0], thre_max[0], steps=self.tau_step)
            tau_grid = torch.linspace(0, 1, steps=self.tau_step + 1)[0:-1]
            ss_diff = torch.zeros((self.n_classes_c, self.n_classes_c), device=preds.device)
            total = torch.zeros(self.n_classes_c, device=preds.device)
            for c_temp in range(self.n_classes_c):
                total[c_temp] = sum(control == c_temp)

            prob_s = total / total.sum()
            for tau in tau_grid:
                count = torch.zeros(self.n_classes_c, device=preds.device)
                temp1 = (preds_n > tau).squeeze(1)
                for c_temp in range(self.n_classes_c):
                    temp2 = (control == c_temp)
                    count[c_temp] += sum(temp1 & temp2)

                # prob is conditional probability, P(Y_hat | S)

                prob = count / torch.clamp(total, 1.0)
                #
                # for c1 in range(self.n_classes_c):
                #     for c2 in range(c1 + 1, self.n_classes_c):
                #         sq_diff = torch.abs(prob[c1] - prob[c2])
                #         ss_diff[c1, c2] += sq_diff / self.tau_step
                sq_s = torch.tile(prob, (len(prob), 1))
                ss_diff += torch.abs(sq_s - sq_s.transpose(1, 0)) / self.tau_step

        if preds_n.shape[1] == 2:
            # tau_grid0 = torch.linspace(thre_min[0], thre_max[0], steps=self.tau_step)
            # tau_grid1 = torch.linspace(thre_min[1], thre_max[1], steps=self.tau_step)

            tau_grid0 = torch.linspace(0, 1, steps=self.tau_step+1)[0:-1]
            tau_grid1 = torch.linspace(0, 1, steps=self.tau_step+1)[0:-1]
            # import pdb; pdb.set_trace()
            ss_diff = torch.zeros((self.n_classes_c, self.n_classes_c), device=preds.device)
            total = torch.zeros(self.n_classes_c, device=preds.device)
            for c_temp in range(self.n_classes_c):
                total[c_temp] = sum(control == c_temp)

            prob_s = total / total.sum()

            for tau0 in tau_grid0:
                temp0 = (preds_n[:, 0] > tau0)
                for tau1 in tau_grid1:
                    # count_y = torch.zeros(1, device=preds.device)
                    count = torch.zeros(self.n_classes_c, device=preds.device)

                    temp1 = temp0 & (preds_n[:, 1] > tau1)

                    # count_y += sum(temp1)
                    for c_temp in range(self.n_classes_c):
                        temp2 = (control == c_temp)
                        count[c_temp] += sum(temp1 & temp2)

                    # prob is conditional probability, P(Y_hat | S)

                    prob = count / torch.clamp(total, 1.0)

                    # for c1 in range(self.n_classes_c):
                    #     for c2 in range(c1 + 1, self.n_classes_c):
                    #         sq_diff += torch.abs(prob[c1] - prob[c2])
                    #         ss_diff[c1, c2] += sq_diff / (self.tau_step**2)

                    sq_s = torch.tile(prob, (len(prob), 1))
                    ss_diff += torch.abs(sq_s - sq_s.transpose(1, 0)) / (self.tau_step**2)

        prob_s = prob_s.unsqueeze(0)
        dp_avg = torch.mm(torch.mm(prob_s, ss_diff), prob_s.t())
        dp_max = ss_diff.max()

        output["DP_avg"] = dp_avg
        output["DP_max"] = dp_max

        return output

class DemographicParityQuantize(tm.Metric):
    """
    DPV when the differences are calculated between the conditional
    probailities for each (y, s, s') triplet.
    """

    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 n_dim_y: int = 3,
                 n_levels_y: int = 10,
                 n_classes_s: int = 2,
                 n_attributes_c: int = None,
                 domind: bool = False,
                 discrete_y=None,
                 discrete_s=None):

        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)

        self.discrete_y = discrete_y
        self.discrete_s = discrete_s

        # self.add_state("total", default=torch.zeros((n_dim_y, n_levels_y, n_classes_c)), dist_reduce_fx=None)
        # self.add_state("count", default=torch.zeros((n_dim_y, n_levels_y, n_classes_c)), dist_reduce_fx=None)
        # self.add_state("count_y", default=torch.zeros(n_dim_y, n_levels_y), dist_reduce_fx=None)
        self.add_state("prediction", default=[], dist_reduce_fx=None)
        self.add_state("control", default=[], dist_reduce_fx=None)

        self.n_levels_y = n_levels_y
        self.n_dim_y = n_dim_y
        self.n_classes_c = n_classes_s
        self.n_attributes_c = n_attributes_c

        if isinstance(domind, str):
            self.domind = True if domind.lower() == "true" else False
        else:
            self.domind = domind

    def update(self, preds, control):

        if self.discrete_s is not None:
            control = discretize(control, self.discrete_s, sens=False)
        else:
            if control.size(1) == self.n_classes_c[0]:
                assert len(self.n_classes_c) == 1
                control = control.argmax(dim=1)

        self.prediction.append(preds)
        self.control.append(control)

    def compute(self):

        preds = torch.cat(self.prediction, 0)
        control = torch.cat(self.control, 0)

        preds_n = (preds - torch.min(preds, dim=0)[0]) / (torch.max(preds, dim=0)[0] - torch.min(preds, dim=0)[0])
        # cont_y = torch.zeros()

        output = DetachableDict()
        dp_avg = 0
        dp_max = 0

        for dim in range(self.n_dim_y):
            max_diff = []
            avg_diff = []

            count_y_q = quantize_eq1(preds_n[:, dim], num=self.n_levels_y)

            count_y = torch.zeros(self.n_levels_y, device=preds.device)
            # total = torch.zeros((self.n_levels_y, self.n_classes_c), device=preds.device)
            total = torch.zeros(self.n_classes_c, device=preds.device)
            count = torch.zeros((self.n_levels_y, self.n_classes_c), device=preds.device)

            for c_temp in range(self.n_classes_c):
                total[c_temp] = sum(control == c_temp)

            for y_temp in range(self.n_levels_y):
                temp1 = (count_y_q == y_temp)
                count_y[y_temp] += sum(temp1)
                for c_temp in range(self.n_classes_c):
                    # total[y_temp, c_temp] += sum(control == c_temp)
                    temp2 = (control == c_temp)
                    count[y_temp, c_temp] += sum(temp1 & temp2)

            # prob is conditional probability, P(Y_hat | S)
            prob = count / torch.clamp(total, 1.0)
            prob_s = total / total.sum()
            prob_s = prob_s.unsqueeze(0)

            # calculate the difference between all pairs of probabilities
            for y_temp in range(self.n_levels_y):
                # max_for_this_tgt_class = 0
                # ss_diff = torch.zeros((self.n_classes_c, self.n_classes_c), device=prob_s.device)
                # for c1 in range(self.n_classes_c):
                #     for c2 in range(c1 + 1, self.n_classes_c):
                #         sq_diff = torch.abs(prob[y_temp, c1] - prob[y_temp, c2])
                #         # sq_diff = torch.abs(prob[y_temp, c1] - prob[y_temp, c2])
                #         if sq_diff > max_for_this_tgt_class:
                #             max_for_this_tgt_class = sq_diff
                #         ss_diff[c1, c2] = sq_diff
                #         # Adding transpose of ss_diff to avoid computing the
                # # same
                # ss_diff += torch.transpose(ss_diff.clone(), 0, 1)

                sq_s = torch.tile(prob[y_temp], (len(prob[y_temp]), 1))
                ss_diff = torch.abs(sq_s - sq_s.transpose(1, 0))
                # import pdb; pdb.set_trace()
                exp_ss_diff = torch.mm(torch.mm(prob_s, ss_diff), prob_s.t())
                max_for_this_tgt_class = ss_diff.max()

                avg_diff.append(exp_ss_diff.item())
                max_diff.append(max_for_this_tgt_class)

            # average the quantities according to the probabilities

            prob_y = count_y / count_y.sum()
            max_diff = torch.tensor(max_diff, device=prob_y.device)
            max_diff = (prob_y * max_diff).sum()
            avg_diff = torch.tensor(avg_diff, device=prob_y.device)
            avg_diff = (prob_y * avg_diff).sum()

            dp_avg += avg_diff / self.n_dim_y
            dp_max += max_diff / self.n_dim_y

        output["DP_avg"] = dp_avg
        output["DP_max"] = dp_max

        return output

class DemographicParity(tm.Metric):
    """
    DPV when the differences are calculated between the conditional
    probailities for each (y, s, s') triplet.
    """
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 n_classes_y: int = 2,
                 n_classes_c: list = [2],
                 n_attributes_c: int = None,
                 domind: bool = False,
                 discrete_y = None,
                 discrete_s = None):

        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)

        self.discrete_y = discrete_y
        self.discrete_s = discrete_s

        # We will keep the option for multiple attributes for sensitive
        # features because DPV is calculated for each sensitive attribute.
        if n_attributes_c is not None:
            if len(n_classes_c) != n_attributes_c:
                n_classes_c = n_classes_c * n_attributes_c
        else:
            n_attributes_c = len(n_classes_c)

        for cntl in range(n_attributes_c):
            self.add_state("total_" + str(cntl), default=torch.zeros((n_classes_y, n_classes_c[cntl])), dist_reduce_fx=None)
            self.add_state("count_" + str(cntl), default=torch.zeros((n_classes_y, n_classes_c[cntl])), dist_reduce_fx=None)
            self.add_state("count_y_" + str(cntl), default=torch.zeros(n_classes_y), dist_reduce_fx=None)

        self.n_classes_y = n_classes_y
        self.n_classes_c = n_classes_c
        self.n_attributes_c = n_attributes_c

        if isinstance(domind, str):
            self.domind = True if domind.lower() == "true" else False
        else:
            self.domind = domind

    def update(self, preds, control):
        if self.discrete_s is not None:
            control = discretize(control, self.discrete_s, sens=False)
        else:
            # We originally assumed that control will be batch of class
            # labels. But we wrote everything else assuming that control
            # will be batch of onehot vectors. So we need to convert the
            # control to class labels from onehot vectors. But in some
            # cases, like FolkTables with age as the sensitive attribute, we
            # still have classs label.
            if len(control.squeeze().shape) > 1:
                if control.size(1) == self.n_classes_c[0]:
                    assert len(self.n_classes_c) == 1, "Multi-class one-hot is not supported yet"
                    control = control.argmax(dim=1)
            else:
                pass
        
        # This is for compatibility with the original code written by
        # professor.
        control = control.reshape(-1, self.n_attributes_c)

        # If we expect the predictions to be discrete, they should be of
        # size (batch_size, 1). Then we have to discretize them
        # according to the discrete levels.
        if self.discrete_y is not None:
            pred = discretize(preds, self.discrete_y)
        # If the predictions are coming from a DomainIndependent model,
        # we need to process it to get the predicted class.
        else:
            if self.domind:
                num_domains = preds.shape[1] // self.n_classes_y
                preds = torch.reshape(preds, (-1, num_domains, self.n_classes_y))
                preds = torch.mean(preds, dim=1)
            
            assert preds.shape[1] == self.n_classes_y
            pred = preds.argmax(dim=1, keepdim=False)

        for cntl in range(self.n_attributes_c):
            for y_temp in range(self.n_classes_y):
                getattr(self, "count_y_" + str(cntl))[y_temp] += sum(pred == y_temp)
                for c_temp in range(self.n_classes_c[cntl]):
                    getattr(self, "total_" + str(cntl))[y_temp, c_temp] += sum(control[:, cntl] == c_temp)
                    temp1 = (pred == y_temp)
                    temp2 = (control[:, cntl] == c_temp)
                    getattr(self, "count_" + str(cntl))[y_temp, c_temp] += sum(temp1 & temp2)

    def compute(self):
        output = DetachableDict()
        for cntl in range(self.n_attributes_c):
            max_diff = []
            avg_diff = []
            # prob is conditional probability, P(Y_hat | S)
            prob = getattr(self, "count_" + str(cntl)) / torch.clamp(getattr(self, "total_" + str(cntl)), 1.0)
            prob_s = getattr(self, "total_" + str(cntl))[0].reshape(1, -1)
            prob_s /= prob_s.sum()

            # calculate the difference between all pairs of probabilities
            for y_temp in range(self.n_classes_y):
                max_for_this_tgt_class = 0
                ss_diff = torch.zeros((self.n_classes_c[cntl], self.n_classes_c[cntl]), device=prob_s.device, dtype=prob_s.dtype)
                for c1 in range(self.n_classes_c[cntl]):
                    for c2 in range(c1+1, self.n_classes_c[cntl]):
                        sq_diff = (prob[y_temp, c1] - prob[y_temp, c2])**2
                        if sq_diff > max_for_this_tgt_class:
                            max_for_this_tgt_class = sq_diff
                        ss_diff[c1, c2] = sq_diff                
                # Adding transpose of ss_diff to avoid computing the
                # same
                ss_diff += torch.transpose(ss_diff.clone(), 0, 1)
                # import pdb; pdb.set_trace()
                exp_ss_diff = torch.mm(torch.mm(prob_s, ss_diff), prob_s.t())
                avg_diff.append(exp_ss_diff.item())
                max_diff.append(max_for_this_tgt_class)

            # average the quantities according to the probabilities
            prob_y = getattr(self, "count_y_" + str(cntl))
            prob_y /= prob_y.sum()
            max_diff = torch.tensor(max_diff, device=prob_y.device)
            max_diff = (prob_y*max_diff).sum()
            avg_diff = torch.tensor(avg_diff, device=prob_y.device)
            avg_diff = (prob_y*avg_diff).sum()

            output["DP_avg_" + str(cntl)] = avg_diff
            output["DP_max_" + str(cntl)] = max_diff

        return output


