import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.model.examplar import ExemplarHandler
import copy
import math

from utils.model.module.classifier import build_classifier


class my_model(ExemplarHandler):
    def __init__(self, backbone=None, pretrained=False, args=None):
        super(my_model, self).__init__()
        self.backbone = backbone(pretrained=pretrained, mode=args.mode)
        self.fc_features = self.backbone.feature_dim
        self.args = args
        self.classifier = build_classifier(args, self.fc_features)
        self.proto_all = nn.ParameterList([])

    def forward(self, query_image):
        features = self.backbone(query_image)
        num_batch = features.shape[0]
        output = self.classifier(features, num_batch)
        return output

    def feature_extractor(self, images):
        features = self.backbone(images)
        return features

    def classify(self, features):
        num_batch = features.shape[0]
        output = self.classifier(features, num_batch)
        return output

    def fix_backbone(self):
        """Freeze the backbone domain-agnostic"""
        for k, v in self.backbone.named_parameters():
            if ('adapter' not in k) and ('cls' not in k) and ('running' not in k):
                v.requires_grad = False

    def fix_backbone_adapter(self):
        """Freeze the backbone domain-agnostic"""
        for k, v in self.backbone.named_parameters():
            if 'adapter' not in k:
                v.requires_grad = False

    def fix_backbone_all(self):
        """Freeze the backbone domain-agnostic"""
        for k, v in self.backbone.named_parameters():
            v.requires_grad = False

    def fuse_backbone(self):
        model_dict = self.model.state_dict()
        for k, v in model_dict.items():
            if 'adapter' in k:
                k_conv3 = k.replace('adapter', 'conv')
                model_dict[k_conv3] = model_dict[k_conv3] + F.pad(v, [1, 1, 1, 1], 'constant', 0)
                model_dict[k] = torch.zeros_like(v)
        self.model.load_state_dict(model_dict)

    def copy(self):
        return copy.deepcopy(self)

    @property
    def proto(self):
        return self.proto_all[-1]

    def train_mode(self):
        self.train()
        self.backbone.eval()

