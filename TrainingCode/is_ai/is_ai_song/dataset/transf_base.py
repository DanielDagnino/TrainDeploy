#!/usr/bin/env python
import random
import numpy as np


class OneOf:
    def __init__(self, transforms_list, p=None):
        self.transforms_list = transforms_list
        self.p = p

    def __call__(self, *args):
        trans = np.random.choice(self.transforms_list, p=self.p)
        args = trans(*args)
        return args


class AllShuffled:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, *args):
        shuffled_transforms_list = random.sample(self.transforms_list, len(self.transforms_list))
        for op in shuffled_transforms_list:
            args = op(*args)
        return args


class AllSequential:
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, *args):
        for op in self.transforms_list:
            args = op(*args)
        return args
