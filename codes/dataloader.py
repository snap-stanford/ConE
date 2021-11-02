#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import os
import json

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import random

class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, degree=None, viable_neg=None):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.degree = degree
        self.viable_neg = viable_neg
        self.warning_flag = True
        self.distribution = None

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        if len(positive_sample) == 3:
            head, relation, tail = positive_sample
            category = None
        elif len(positive_sample) == 4:
            head, relation, tail, category = positive_sample
        else:
            raise ValueError('The length of triple must be either 3 or 4')

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation-1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))
        
        negative_sample_list = []
        negative_sample_size = 0

        if self.viable_neg is not None:
            # Faster negative sampling based on distribution
            negative_sample_list = []
            negative_sample_size = 0

            while negative_sample_size < self.negative_sample_size:

                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 1000)

                array = np.array(list(self.viable_neg[str(head)]))

                if self.mode == 'head-batch':
                    raise NotImplementedError
                elif self.mode == 'tail-batch':
                    mask = np.in1d(
                        negative_sample,
                        array,
                        assume_unique=False,
                        invert=True
                    )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size

            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            negative_sample = torch.from_numpy(negative_sample)

        else:
            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
                if self.mode == 'head-batch':
                    if self.viable_neg is None:
                        mask = np.in1d(
                            negative_sample,
                            self.true_head[(relation, tail)],
                            assume_unique=True,
                            invert=True
                        )
                    else:
                        raise NotImplementedError
                elif self.mode == 'tail-batch':
                    if self.viable_neg is None:
                        # raise ValueError('as a precaution for not selecting negative samples')
                        mask = np.in1d(
                            negative_sample,
                            self.true_tail[(head, relation)],
                            assume_unique=True,
                            invert=True
                        )
                    else:
                        # Warning: viable_neg here is actually the complement of viable_neg

                        mask = np.in1d(
                            negative_sample,
                            np.array(list(self.viable_neg[str(head)])),
                            assume_unique=False,
                            invert=True
                        )
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample = negative_sample[mask]
                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size
        
            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            negative_sample = torch.from_numpy(negative_sample)

        if len(positive_sample) == 3:
            positive_sample = torch.LongTensor((head, relation, tail))
        else:
            positive_sample = torch.LongTensor((head, relation, tail, category))

        if self.degree is None:
            return positive_sample, negative_sample, subsampling_weight, self.mode
        else:
            if self.mode == 'head-batch':
                raise NotImplementedError
            deg = torch.Tensor([self.degree[(head, relation)]])
            return positive_sample, negative_sample, subsampling_weight, self.mode, deg

    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        if len(data[0]) == 5:
            degree = torch.stack([_[4] for _ in data], dim=0)
            return positive_sample, negative_sample, subsample_weight, mode, degree
        else:
            return positive_sample, negative_sample, subsample_weight, mode


    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''

        if len(triples[0]) == 4:
            # adjust for the case in which triples include additional relation category label
            count = {}
            for head, relation, tail, _ in triples:
                if (head, relation) not in count:
                    count[(head, relation)] = start
                else:
                    count[(head, relation)] += 1

                if (tail, -relation - 1) not in count:
                    count[(tail, -relation - 1)] = start
                else:
                    count[(tail, -relation - 1)] += 1
        else:
            count = {}
            for head, relation, tail in triples:
                if (head, relation) not in count:
                    count[(head, relation)] = start
                else:
                    count[(head, relation)] += 1

                if (tail, -relation-1) not in count:
                    count[(tail, -relation-1)] = start
                else:
                    count[(tail, -relation-1)] += 1
        return count
    
    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''
        
        true_head = {}
        true_tail = {}

        if len(triples[0]) == 4:
            # adjust for the case in which triples include additional relation category label
            for head, relation, tail, _ in triples:
                if (head, relation) not in true_tail:
                    true_tail[(head, relation)] = []
                true_tail[(head, relation)].append(tail)
                if (relation, tail) not in true_head:
                    true_head[(relation, tail)] = []
                true_head[(relation, tail)].append(head)
        else:
            for head, relation, tail in triples:
                if (head, relation) not in true_tail:
                    true_tail[(head, relation)] = []
                true_tail[(head, relation)].append(tail)
                if (relation, tail) not in true_head:
                    true_head[(relation, tail)] = []
                true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TrainDatasetDualNegative(TrainDataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, degree=None, viable_neg=None):
        super(TrainDatasetDualNegative, self).__init__(triples, nentity, nrelation, negative_sample_size, mode,
                                                       degree,
                                                       viable_neg)

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        if len(positive_sample) == 3:
            head, relation, tail = positive_sample
        elif len(positive_sample) == 4:
            head, relation, tail, category = positive_sample
        else:
            raise ValueError('The length of triple must be either 3 or 4')

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        if not category == 0:  # hierarchical relationship
            p = self.distribution[str(head)]
            negative_sample = np.random.choice(self.nentity, size=self.negative_sample_size, p=p)
            negative_sample = torch.from_numpy(negative_sample)
        else:
            negative_sample_list = []
            negative_sample_size = 0

            while negative_sample_size < self.negative_sample_size:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
                if self.mode == 'head-batch':
                    raise NotImplementedError
                elif self.mode == 'tail-batch':
                    mask = np.in1d(
                        negative_sample,
                        self.true_tail[(head, relation)],
                        assume_unique=True,
                        invert=True
                    )

                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample = negative_sample[mask]

                negative_sample_list.append(negative_sample)
                negative_sample_size += negative_sample.size

            negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
            negative_sample = torch.from_numpy(negative_sample)

        if len(positive_sample) == 3:
            positive_sample = torch.LongTensor((head, relation, tail))
        else:
            positive_sample = torch.LongTensor((head, relation, tail, category))

        if self.degree is None:
            return positive_sample, negative_sample, subsampling_weight, self.mode
        else:
            raise NotImplementedError
            deg = torch.Tensor([self.degree[(head, relation)]])
            return positive_sample, negative_sample, subsampling_weight, self.mode, deg

class ClassDataset(Dataset):
    def __init__(self, triples):
        self.len = len(triples)
        self.triples = triples
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        sample = self.triples[idx]
        if len(sample) == 4:
            h, r, t, l = sample
            sample = torch.LongTensor((h, r, t, l))
        elif len(sample) == 5:
            h, r, t, l, c = sample
            sample = torch.LongTensor((h, r, t, l, c))
        return sample
    
    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        l = torch.stack([_[3] for _ in data], dim=0)
        if len(data[0]) == 4:
            return h, r, t, l
        elif len(data[0]) == 5:
            c = torch.stack([_[4] for _ in data], dim=0)
            return h, r, t, l, c
    
    @staticmethod
    def collate_fn(data):
        h = torch.stack([_[0] for _ in data], dim=0)
        r = torch.stack([_[1] for _ in data], dim=0)
        t = torch.stack([_[2] for _ in data], dim=0)
        l = torch.stack([_[3] for _ in data], dim=0)
        if len(data[0]) == 4:
            return h, r, t, l
        elif len(data[0]) == 5:
            c = torch.stack([_[4] for _ in data], dim=0)
            return h, r, t, l, c

class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode, degree=None):
        self.len = len(triples)

        temp = []
        for triple in all_true_triples:
            if len(triple) == 4:
                triple = triple[0:3]
            temp.append(triple)
        all_true_triples = temp

        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.degree = degree

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        if len(positive_sample) == 3:
            head, relation, tail = positive_sample
            category = None
        elif len(positive_sample) == 4:
            head, relation, tail, category = positive_sample
        else:
            raise ValueError('The length of triple must be either 3 or 4')

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)
            
        tmp = torch.LongTensor(tmp)            
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        if len(positive_sample) == 3:
            positive_sample = torch.LongTensor((head, relation, tail))
        else:
            positive_sample = torch.LongTensor((head, relation, tail, category))

        if self.degree is None:
            return positive_sample, negative_sample, filter_bias, self.mode
        else:
            if self.mode == 'head-batch':
                raise NotImplementedError
            deg = torch.Tensor([self.degree[(head, relation)]])
            return positive_sample, negative_sample, filter_bias, self.mode, deg

    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        if len(data[0]) == 5:
            degree = torch.stack([_[4] for _ in data], dim=0)
            return positive_sample, negative_sample, filter_bias, mode, degree
        else:
            return positive_sample, negative_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class UnidirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        self.dataloader = dataloader

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data

