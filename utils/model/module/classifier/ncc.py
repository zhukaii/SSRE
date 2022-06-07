import torch.nn as nn
import torch


class NCCClassifier(nn.Module):
    def __init__(self, outplanes, args):
        super(NCCClassifier, self).__init__()
        self.args = args
        self.num_classes = args.num_classes
        self.scale = nn.Parameter(torch.tensor(10.0), requires_grad=True)

    def forward(self, x, num_batch):
        query = x[:num_batch]
        proto = x[num_batch:]
        query = torch.nn.functional.normalize(query, p=2, dim=-1, eps=1e-12)
        proto = torch.nn.functional.normalize(proto, p=2, dim=0, eps=1e-12)
        cos_dist = query @ proto
        scores = self.scale * cos_dist
        return scores
