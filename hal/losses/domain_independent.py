# domain_independent.py

import torch
from torch import nn

__all__ = ['DomainIndependentClassification']

class DomainIndependentClassification(nn.Module):
    def __init__(self, num_domains, num_classes):
        """
        Domain independent loss was proposed in the paper:
        "Towards Fairness in Visual Recognition"

        This class calculates cross entropy loss when y is provided as
        one-hot encoding instead of class index.
        """
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.num_domains = num_domains
        self.num_classes = num_classes

    def forward(self, pred, tgt, sens):
        """
        Here, pred is of dimension (batch_size,
        num_classes*num_domains). tgt is of dimension (batch_size,
        num_classes). We need to convert tgt to (batch_size,
        num_classes*num_domains) using sens
        """
        if sens.size(1) == self.num_domains:
            sens = torch.argmax(sens, dim=1)
        
        sens = sens.reshape(-1)

        assert pred.size(1) == self.num_classes * self.num_domains

        pred_log_softmax = []
        for i in range(self.num_domains):
            pred_log_softmax.append(self.log_softmax(pred[:, i*self.num_classes:(i+1)*self.num_classes]))
        pred_log_softmax = torch.cat(pred_log_softmax, dim=1)

        tgt_class = sens*self.num_classes + torch.argmax(tgt, 1)
        tgt_class = tgt_class.type(torch.long).to(pred.device)

        return self.nll(pred_log_softmax, tgt_class)
