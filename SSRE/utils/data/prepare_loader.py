from torch.utils.data import DataLoader
import torch
import random
import numpy as np


class CategoriesSampler:

    def __init__(self, index, lenth, way, shot):
        self.lenth = lenth
        self.way = way
        self.shot = shot

        self.index = index

    def __len__(self):
        return self.lenth

    def __iter__(self):
        for i_batch in range(self.lenth):
            batch = []
            classes = list(self.index.keys())
            for c in classes:
                lenth_per = torch.from_numpy(self.index[c])
                way_per = len(lenth_per)
                shot = torch.randperm(way_per)[:self.shot]
                batch.append((c * way_per + shot).int())
            batch = torch.stack(batch).reshape(-1)
            yield batch


def prepare_loader_normal(dataset, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              pin_memory=True)
    return loader


def prepare_loader_sample1(dataset, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        pin_memory=True)
    train_sampler = CategoriesSampler(dataset.sub_indexes,
                                      len(loader),
                                      args.way + 3,
                                      args.shot)
    train_fsl_loader = DataLoader(dataset=dataset,
                                  batch_sampler=train_sampler,
                                  num_workers=4,
                                  pin_memory=True)
    return loader, train_fsl_loader


def prepare_loader_sample2(dataset, args):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    train_sampler = CategoriesSampler(dataset.sub_indexes,
                                      len(loader),
                                      max(args.way, args.base_class),
                                      args.shot)
    train_fsl_loader = DataLoader(dataset=dataset,
                                  batch_sampler=train_sampler,
                                  num_workers=4,
                                  pin_memory=True)
    return loader, train_fsl_loader


