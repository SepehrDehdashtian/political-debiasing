
class DemographicParityCEConf(tm.Metric):
    """
    DPV when the differences are calculated between the conditional
    probailities for each (y, s, s') triplet.
    """

    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 n_classes_y: int = 4,
                 top_y: int = 1,
                 n_classes_c: list = [2],
                 n_attributes_c: int = None,
                 domind: bool = False,
                 discrete_s=None):

        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)

        self.discrete_s = discrete_s

        self.add_state("total", default=torch.zeros((n_classes_y, n_classes_c)), dist_reduce_fx=None)
        self.add_state("count", default=torch.zeros((n_classes_y, n_classes_c)), dist_reduce_fx=None)
        self.add_state("count_y", default=torch.zeros(n_classes_y), dist_reduce_fx=None)

        self.n_classes_y = n_classes_y
        self.top_y = top_y
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
            if control.size(1) == self.n_classes_c[0]:
                assert len(self.n_classes_c) == 1, "Multi-class one-hot is not supported yet"
                control = control.argmax(dim=1)

        # This is for compatibility with the original code written by
        # professor.
        pred_y = torch.topk(preds, self.top_y, dim=1).indices
        # pred_y0 = torch.argmax(preds, dim=1)
        # pred_y = preds

        for y_temp in np.ndindex((self.n_classes_y, self.n_classes_y)):
            import pdb; pdb.set_trace()
            temp1 = (pred_y == y_temp)
            getattr(self, "count_y")[y_temp] += sum(temp1)
            for c_temp in range(self.n_classes_c):
                getattr(self, "total")[y_temp, c_temp] += sum(control == c_temp)
                temp2 = (control == c_temp)
                getattr(self, "count")[y_temp, c_temp] += sum(temp1 & temp2)

        # import pdb; pdb.set_trace()

    def compute(self):
        output = DetachableDict()
        dp_avg = 0
        dp_max = 0
        for top in range(self.top_y):
            max_diff = []
            avg_diff = []
            # prob is conditional probability, P(Y_hat | S)
            prob = getattr(self, "count")[top] / torch.clamp(getattr(self, "total")[top], 1.0)
            prob_s = getattr(self, "total")[top][0].reshape(1, -1)
            prob_s /= prob_s.sum()

            # calculate the difference between all pairs of probabilities
            for y_temp in range(self.n_classes_y):
                max_for_this_tgt_class = 0
                ss_diff = torch.zeros((self.n_classes_c, self.n_classes_c), device=prob_s.device)
                for c1 in range(self.n_classes_c):
                    for c2 in range(c1 + 1, self.n_classes_c):
                        sq_diff = (prob[y_temp, c1] - prob[y_temp, c2]) ** 2
                        if sq_diff > max_for_this_tgt_class:
                            max_for_this_tgt_class = sq_diff
                        ss_diff[c1, c2] = sq_diff
                        # Adding transpose of ss_diff to avoid computing the
                # same
                ss_diff += torch.transpose(ss_diff.clone(), 0, 1)
                exp_ss_diff = torch.mm(torch.mm(prob_s, ss_diff), prob_s.t())
                avg_diff.append(exp_ss_diff.item())
                max_diff.append(max_for_this_tgt_class)

            # average the quantities according to the probabilities
            prob_y = getattr(self, "count_y")[top]
            prob_y /= prob_y.sum()
            max_diff = torch.tensor(max_diff, device=prob_y.device)
            max_diff = (prob_y * max_diff).sum()
            avg_diff = torch.tensor(avg_diff, device=prob_y.device)
            avg_diff = (prob_y * avg_diff).sum()

            dp_avg += avg_diff / self.top_y
            dp_max += max_diff / self.top_y

        output["DP_avg"] = torch.sqrt(dp_avg)
        output["DP_max"] = torch.sqrt(dp_max)

        return output

class DemographicParityCE(tm.Metric):
    """
    DPV when the differences are calculated between the conditional
    probailities for each (y, s, s') triplet.
    """

    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 n_classes_y: int = 4,
                 top_y: int = 1,
                 n_classes_c: list = [2],
                 n_attributes_c: int = None,
                 domind: bool = False,
                 discrete_s=None):

        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         dist_sync_fn=dist_sync_fn)

        self.discrete_s = discrete_s

        self.add_state("total", default=torch.zeros((top_y, n_classes_y, n_classes_c)), dist_reduce_fx=None)
        self.add_state("count", default=torch.zeros((top_y, n_classes_y, n_classes_c)), dist_reduce_fx=None)
        self.add_state("count_y", default=torch.zeros(top_y, n_classes_y), dist_reduce_fx=None)

        self.n_classes_y = n_classes_y
        self.top_y = top_y
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
            if control.size(1) == self.n_classes_c[0]:
                assert len(self.n_classes_c) == 1, "Multi-class one-hot is not supported yet"
                control = control.argmax(dim=1)

        # This is for compatibility with the original code written by
        # professor.
        pred_y = torch.topk(preds, self.top_y, dim=1).indices
        # pred_y0 = torch.argmax(preds, dim=1)
        # pred_y = preds
        for top in range(self.top_y):
            for y_temp in range(self.n_classes_y):
                temp1 = (pred_y[:, top] == y_temp)
                getattr(self, "count_y")[top, y_temp] += sum(temp1)
                for c_temp in range(self.n_classes_c):
                    getattr(self, "total")[top, y_temp, c_temp] += sum(control == c_temp)
                    temp2 = (control == c_temp)
                    getattr(self, "count")[top, y_temp, c_temp] += sum(temp1 & temp2)

        # import pdb; pdb.set_trace()

    def compute(self):
        output = DetachableDict()
        dp_avg = 0
        dp_max = 0
        for top in range(self.top_y):
            max_diff = []
            avg_diff = []
            # prob is conditional probability, P(Y_hat | S)
            prob = getattr(self, "count")[top] / torch.clamp(getattr(self, "total")[top], 1.0)
            prob_s = getattr(self, "total")[top][0].reshape(1, -1)
            prob_s /= prob_s.sum()

            # calculate the difference between all pairs of probabilities
            for y_temp in range(self.n_classes_y):
                max_for_this_tgt_class = 0
                ss_diff = torch.zeros((self.n_classes_c, self.n_classes_c), device=prob_s.device)
                for c1 in range(self.n_classes_c):
                    for c2 in range(c1 + 1, self.n_classes_c):
                        sq_diff = (prob[y_temp, c1] - prob[y_temp, c2]) ** 2
                        if sq_diff > max_for_this_tgt_class:
                            max_for_this_tgt_class = sq_diff
                        ss_diff[c1, c2] = sq_diff
                        # Adding transpose of ss_diff to avoid computing the
                # same
                ss_diff += torch.transpose(ss_diff.clone(), 0, 1)
                exp_ss_diff = torch.mm(torch.mm(prob_s, ss_diff), prob_s.t())
                avg_diff.append(exp_ss_diff.item())
                max_diff.append(max_for_this_tgt_class)

            # average the quantities according to the probabilities
            prob_y = getattr(self, "count_y")[top]
            prob_y /= prob_y.sum()
            max_diff = torch.tensor(max_diff, device=prob_y.device)
            max_diff = (prob_y * max_diff).sum()
            avg_diff = torch.tensor(avg_diff, device=prob_y.device)
            avg_diff = (prob_y * avg_diff).sum()

            dp_avg += avg_diff / self.top_y
            dp_max += max_diff / self.top_y

        output["DP_avg"] = torch.sqrt(dp_avg)
        output["DP_max"] = torch.sqrt(dp_max)

        return output

class DemographicParityMarginal(tm.Metric):
    """
    DPV when the differences are calculated between the conditional
    probailities and the marginal probabilities for each y, s pair.
    """
    def __init__(self,
                 compute_on_step: bool = False,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 dist_sync_fn: Callable = None,
                 n_classes_y: int = 2,
                 n_classes_c: list = [2],
                 n_attributes_c: int = 2,
                 domind: bool = False):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if len(n_classes_c) != n_attributes_c:
            n_classes_c = n_classes_c * n_attributes_c

        for cntl in range(n_attributes_c):
            self.add_state("total_" + str(cntl), default=torch.zeros((n_classes_y, n_classes_c[cntl])), dist_reduce_fx=None)
            self.add_state("count_" + str(cntl), default=torch.zeros((n_classes_y, n_classes_c[cntl])), dist_reduce_fx=None)

        self.n_classes_y = n_classes_y
        self.n_classes_c = n_classes_c
        self.n_attributes_c = n_attributes_c

        if isinstance(domind, str):
            self.domind = True if domind.lower() == "true" else False
        else:
            self.domind = domind

    def update(self, preds, control):
        # We originally assumed that control will be batch of class
        # labels. But we wrote everything else assuming that control
        # will be batch of onehot vectors. So we need to convert the
        # control to class labels from onehot vectors.
        control = control.argmax(dim=1).reshape(-1, 1)
        assert control.shape[1] == self.n_attributes_c

        """
        If the predictions are coming from a DomainIndependent model, we
        need to process it to get the predicted class.
        """

        if self.domind:
            num_domains = preds.shape[1] // self.n_classes_y
            preds = torch.reshape(preds, (-1, num_domains, self.n_classes_y))
            preds = torch.mean(preds, dim=1)

        # pred = preds.max(1)[1]
        pred = preds.argmax(dim=1, keepdim=False)
        for cntl in range(self.n_attributes_c):
            for y_temp in range(self.n_classes_y):
                for c_temp in range(self.n_classes_c[cntl]):
                    getattr(self, "total_" + str(cntl))[y_temp, c_temp] += sum(control[:, cntl] == c_temp)
                    getattr(self, "count_" + str(cntl))[y_temp, c_temp] += sum((pred == y_temp) & (control[:, cntl] == c_temp))

    def compute(self):
        output = DetachableDict()
        for cntl in range(self.n_attributes_c):
            max_diff = []
            avg_diff = []
            # prob is conditional probability, P(Y_hat | S)
            prob = getattr(self, "count_" + str(cntl)) / torch.clamp(getattr(self, "total_" + str(cntl)), 1.0)
            # Calculate the joint probability first
            joint_prob = getattr(self, "count_" + str(cntl))/getattr(self, "count_" + str(cntl)).sum()
            # Calculate the marginal probability from joint_prob
            marginal_prob = joint_prob.sum(dim=1)

            # calculate the difference between all pairs of probabilities
            for y_temp in range(self.n_classes_y):
                max_for_this_tgt_class = 0
                for c1 in range(self.n_classes_c[cntl]):
                    sq_diff = (prob[y_temp, c1] - marginal_prob[y_temp])**2
                    if sq_diff > max_for_this_tgt_class:
                        max_for_this_tgt_class = sq_diff
                    avg_diff.append(sq_diff)
                max_diff.append(max_for_this_tgt_class)

            output["DP_avg_" + str(cntl)] = torch.sqrt(torch.Tensor(avg_diff).sum()/(self.n_classes_y*self.n_classes_c[cntl]))
            output["DP_max_" + str(cntl)] = torch.sqrt(torch.Tensor(max_diff).sum()/self.n_classes_y)

        return output