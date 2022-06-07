import torch.nn as nn
import torch

import numpy as np


class NCEClassifier(nn.Module):
    def __init__(self, outplanes, args):
        super(NCEClassifier, self).__init__()
        self.args = args
        self.num_classes = args.base_class
        self.temperature = nn.Parameter(torch.tensor(64.0), requires_grad=True)

    def forward(self, x, num_batch):
        query = x[:num_batch]
        proto = x[num_batch:]
        query = query.unsqueeze(1)
        proto = proto.unsqueeze(0).repeat(num_batch, 1, 1)
        scores = - torch.sum((proto - query) ** 2, 2) / self.temperature
        return scores
