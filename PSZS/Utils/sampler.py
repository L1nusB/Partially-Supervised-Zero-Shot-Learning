import copy
import random
from collections import defaultdict
from typing import Optional
import numpy as np
from torch.utils.data.sampler import Sampler

from PSZS.datasets import CustomDataset

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    
    Code adapted from:
        `<https://github.com/phucty/Deep_metric>`
        `<https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/losses/hard_mine_triplet_loss.py>`
    """
    def __init__(self, dataset: CustomDataset, 
                 num_labels: Optional[int] = None, 
                 num_instances: Optional[int] = None, 
                 batch_size: Optional[int] = None, 
                 max_iters: Optional[int] = None,):
        """Randomly sample N identities, then for each identity,
        randomly sample K instances, therefore batch size is N*K.

        Args:
            dataset (CustomDataset): 
                Dataset to sample from.
            num_labels (Optional[int], optional): 
                Number of different labels per identity in a batch. If not given will be set dynamically 
                based on `num_instances` and `batch_size` and internal default of `16`. Defaults to None.
            num_instances (Optional[int], optional): 
                Number of instances per identity in a batch. If not given will be set dynamically 
                based on `num_instances` and `batch_size` and internal default of `4`. Defaults to None.
            batch_size (Optional[int], optional): 
                Desired batch size. Will be adjusted if does not match `num_instances` and `num_labels` 
                or set as their product if not specified. Defaults to None.
            max_iters (Optional[int], optional): 
                Specific number of iterations the sampler will run for. Defaults to None.
        """
        self.label_index_dict = dataset.label_index
        if batch_size is None:
            # Set default values
            # not in constructor to allow flexibility when batch size is given
            num_labels = num_labels or 16
            num_instances = num_instances or 4
            batch_size = num_labels * num_instances
        else:
            if num_labels is None:
                num_instances = num_instances or 4
                num_labels = round(batch_size / num_instances)
                if batch_size % num_instances != 0:
                    print(f'Batch size {batch_size} is not a multiple of num_instances {num_instances}. '
                            'Setting batch size to the nearest multiple.')
                batch_size = num_labels * num_instances
            elif num_instances is None:
                num_labels = num_labels or 16
                num_labels = round(batch_size / num_labels)
                if batch_size % num_labels != 0:
                    print(f'Batch size {batch_size} is not a multiple of num_labels {num_labels}. '
                            'Setting batch size to the nearest multiple.')
                batch_size = num_labels * num_instances
            else:
                if batch_size != num_labels * num_instances:
                    print(f'Batch size does not match given/default num_instances {num_instances} and num_labels {num_labels}. '
                          f'{batch_size}!={num_instances*num_labels}. Setting batch size to {num_labels * num_instances}.')
                    batch_size = num_labels * num_instances
        self.num_labels_per_batch = num_labels
        self.num_instances_per_label = num_instances
        self.batch_size = batch_size
        print(f'Batch size: {self.batch_size}, num_labels: {self.num_labels_per_batch}, num_instances: {self.num_instances_per_label}')
        
        if max_iters is None:
            self.max_iters = len(dataset) // self.batch_size
        else:
            self.max_iters = max_iters
        self.labels = list(self.label_index_dict.keys())

    def __len__(self):
        return self.max_iters

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"|Sampler| iters {self.max_iters}| K {self.num_instances_per_label}| M {self.num_labels_per_batch}|"

    def _prepare_batch(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.label_index_dict[label])
            if len(idxs) < self.num_instances_per_label:
                idxs.extend(np.random.choice(idxs, size=self.num_instances_per_label - len(idxs), replace=True))
            random.shuffle(idxs)

            batch_idxs_dict[label] = [idxs[i * self.num_instances_per_label: (i + 1) * self.num_instances_per_label] for i in range(len(idxs) // self.num_instances_per_label)]

        avai_labels = copy.deepcopy(self.labels)
        return batch_idxs_dict, avai_labels

    def __iter__(self):
        batch_idxs_dict, avai_labels = self._prepare_batch()
        for _ in range(self.max_iters):
            batch = []
            if len(avai_labels) < self.num_labels_per_batch:
                batch_idxs_dict, avai_labels = self._prepare_batch()

            selected_labels = random.sample(avai_labels, self.num_labels_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                batch.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)
            yield batch
