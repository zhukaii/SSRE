import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, outplanes, args):
        super(LinearClassifier, self).__init__()
        self.cls_fn = nn.Linear(outplanes, args.num_classes)

    def forward(self, x, num_batch):
        x = self.cls_fn(x)
        return x
