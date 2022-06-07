import torch.nn as nn
import torch

import numpy as np


class CosineClassifier(nn.Module):
    def __init__(self, outplanes, args):
        super(CosineClassifier, self).__init__()
        self.num_classes = args.num_classes
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        weight = torch.FloatTensor(outplanes, args.num_classes).normal_(
                    0.0, np.sqrt(2.0 / args.num_classes))
        self.weight = nn.Parameter(weight, requires_grad=True)

    def forward(self, x, num_batch):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-12)
        weight = torch.nn.functional.normalize(self.weight, p=2, dim=0, eps=1e-12)
        cos_dist = x_norm @ weight
        scores = self.scale * cos_dist
        return scores
