# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import itertools

import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import Sampler
from wetectron.data.datasets import PascalVOCDataset
from collections import Counter

class ClassBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.

    Arguments:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_uneven (bool): If ``True``, the sampler will drop the batches whose
            size is less than ``batch_size``

    """

    def __init__(self, sampler, group_ids, batch_size, dataset, args_t, args_v,
                 drop_uneven=False):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = torch.as_tensor(group_ids)
        assert self.group_ids.dim() == 1
        self.batch_size = batch_size
        self.drop_uneven = drop_uneven

        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False

        self.voc_train = PascalVOCDataset(**args_t)
        self.voc_val = PascalVOCDataset(**args_v)
        self.dataset = dataset

    def get_img_labels(self, index):
        if self.dataset.get_idxs(index)[0] == 0:
             img_labels = self.voc_train.get_groundtruth(
                     self.dataset.get_idxs(index)[1]).get_field('labels')
        else :
             img_labels = self.voc_val.get_groundtruth(
                     self.dataset.get_idxs(index)[1]).get_field('labels')
        return img_labels

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        # get the sampled indices from the sampler
        sampled_ids = torch.as_tensor(list(self.sampler))
        # potentially not all elements of the dataset were sampled
        # by the sampler (e.g., DistributedSampler).
        # construct a tensor which contains -1 if the element was
        # not sampled, and a non-negative number indicating the
        # order where the element was sampled.
        # for example. if sampled_ids = [3, 1] and dataset_size = 5,
        # the order is [-1, 1, -1, 0, -1]
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        # get a mask with the elements that were sampled
        mask = order >= 0

        # find the elements that belong to each individual cluster
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        # get relative order of the elements inside each cluster
        # that follows the order from the sampler
        relative_order = [order[cluster] for cluster in clusters]
        # with the relative order, find the absolute order in the
        # sampled space
        permutation_ids = [s[s.sort()[1]] for s in relative_order]
        # permute each cluster so that they follow the order from
        # the sampler
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        # splits each cluster in batch_size, and merge as a list of tensors
        splits = [c.split(self.batch_size) for c in permuted_clusters]
        merged = tuple(itertools.chain.from_iterable(splits))

        # now each batch internally has the right order, but
        # they are grouped by clusters. Find the permutation between
        # different batches that brings them as close as possible to
        # the order that we have in the sampler. For that, we will consider the
        # ordering as coming from the first element of each batch, and sort
        # correspondingly
        first_element_of_batch = [t[0].item() for t in merged]
        # get and inverse mapping from sampled indices and the position where
        # they occur (as returned by the sampler)
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        # from the first element in each batch, get a relative ordering
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )

        # permute the batches so that they approximately follow the order
        # from the sampler
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        # finally, permute thebatches
        batches = [merged[i].tolist() for i in permutation_order]

        if self.drop_uneven:
            kept = []
            for batch in batches:
                if len(batch) == self.batch_size:
                    kept.append(batch)
            batches = kept

        ###

        sampled_ids = sampled_ids.tolist()
        ids = sampled_ids
        c_batches = []
        img_labels = []
        img_labels = [0] * len(sampled_ids) * self.batch_size

        for i in sampled_ids:
            #img_labels.append(self.get_img_labels(i).tolist())
            img_labels[i] = self.get_img_labels(i).tolist()

        for single in sampled_ids:
            if len(set(img_labels[single])) == 1:
                for multi in sampled_ids:
                    if (set(img_labels[single]) & set(img_labels[multi])) and len(set(img_labels[multi])) > 1:
                    #if set(self.get_img_labels(single)) & set(self.get_img_labels(multi)) and len(set(self.get_img_labels(multi))) > 1:
                        c_batches.append([single,multi])
                        sampled_ids.remove(multi)
                        break

        f_batches = list(itertools.chain(*batches))
        f_c_batches = list(itertools.chain(*c_batches))
        remain_ids = list((Counter(f_batches) - Counter(f_c_batches)).elements())

        for inter_1 in remain_ids:
            for inter_2 in remain_ids:
                if set(img_labels[inter_1]) & set(img_labels[inter_2]) and (inter_1 != inter_2):
                    c_batches.append([inter_1,inter_2])
                    remain_ids.remove(inter_2)
                    break
        ###

        return c_batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)

    def get_img_labels(self, index):
        if self.dataset.get_idxs(index)[0] == 0:
             img_labels = self.voc_train.get_groundtruth(
                     self.dataset.get_idxs(index)[1]).get_field('labels')
        else :
             img_labels = self.voc_val.get_groundtruth(
                     self.dataset.get_idxs(index)[1]).get_field('labels')
        return img_labels

    def same_class_single(self, sampled_ids, img_labels):
        c_batches = []
        for single in sampled_ids:
            if len(set(img_labels[single])) == 1:
                for multi in sampled_ids:
                    if (set(img_labels[single]) & set(img_labels[multi])) and len(set(img_labels[multi])) > 1:
                        c_batches.append([single,multi])
                        sampled_ids.remove(single)
                        sampled_ids.remove(multi)
                        break
        return c_batches
