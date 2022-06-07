import torch.nn as nn
import torch

import numpy as np


class EUCClassifier(nn.Module):
    def __init__(self, outplanes, args):
        super(EUCClassifier, self).__init__()
        self.num_classes = args.num_classes
        self.temperature = 64
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        weight = torch.FloatTensor(outplanes, args.num_classes).normal_(
                    0.0, np.sqrt(2.0 / args.num_classes))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, num_batch):
        num_batch = x.shape[0]
        self.weight = self.weight.unsqueeze(0).repeat(num_batch, 1, 1)
        scores = - torch.sum((self.weight - x) ** 2, 2) / self.temperature
        return scores
