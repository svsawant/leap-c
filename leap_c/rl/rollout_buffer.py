import collections
import random
from enum import Enum
from typing import Any
from collections import defaultdict
from copy import deepcopy

import torch
from torch.utils.data._utils.collate import collate
from gymnasium.spaces import Box
import numpy as np

from leap_c.collate import create_collate_fn_map, pytree_tensor_to


# class RolloutBuffer:
#     def __init__(
#         self,
#         buffer_limit: int,
#         device: str,
#         tensor_dtype: torch.dtype = torch.float32,
#     ):
#         """
#         Args:
#             buffer_limit: The maximum number of transitions that can be stored in the buffer.
#                 If the buffer is full, the oldest transitions are discarded when putting in a new one.
#             device: The device to which all sampled tensors will be cast.
#             collate_fn_map: The collate function map that informs the buffer how to form batches.
#             tensor_dtype: The data type to which the tensors in the observation will be cast.
#             input_transformation: A function that transforms the data before it is put into the buffer.
#         """
#         self.buffer = collections.deque(maxlen=buffer_limit)
#         self.device = device
#         self.tensor_dtype = tensor_dtype
#
#         # TODO (Jasper): This should be derived from task.
#         self.collate_fn_map = create_collate_fn_map()
#
#     def put(self, data: Any):
#         """Put the data into the replay buffer. If the buffer is full, the oldest data is discarded.
#
#         Parameters:
#             data: The data to put into the buffer.
#                 It should be collatable according to the custom_collate function.
#         """
#         self.buffer.append(data)
#
#     def sample(self, n: int) -> Any:
#         """
#         Sample a mini-batch from the replay buffer,
#         collate the mini-batch according to self.custom_collate_map
#         and cast all tensors in the collated mini-batch (must be a pytree structure)
#         to the device and dtype of the buffer.
#
#         Parameters:
#             n: The number of samples to draw.
#         """
#         mini_batch = random.sample(self.buffer, n)
#         return pytree_tensor_to(
#             collate(mini_batch, collate_fn_map=self.collate_fn_map),
#             device=self.device,
#             tensor_dtype=self.tensor_dtype,
#         )
#
#     def __len__(self):
#         return len(self.buffer)


class RolloutBuffer(object):
    '''Storage for a batch of episodes during training.

    Attributes:
        max_length (int): maximum length of episode.
        batch_size (int): number of episodes per batch.
        scheme (dict): describes shape & other info of data to be stored.
        keys (list): names of all data from scheme.
    '''

    def __init__(self,
                 obs_space,
                 act_space,
                 max_length,
                 batch_size
                 ):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        T, N = max_length, batch_size
        obs_dim = obs_space.shape
        if isinstance(act_space, Box):
            act_dim = act_space.shape[0]
        else:
            act_dim = act_space.n
        self.scheme = {
            'obs': {
                'vshape': (T, N, *obs_dim)
            },
            'act': {
                'vshape': (T, N, act_dim)
            },
            'rew': {
                'vshape': (T, N, 1)
            },
            'mask': {
                'vshape': (T, N, 1),
                'init': np.ones
            },
            'v': {
                'vshape': (T, N, 1)
            },
            'logp': {
                'vshape': (T, N, 1)
            },
            'ret': {
                'vshape': (T, N, 1)
            },
            'adv': {
                'vshape': (T, N, 1)
            },
            'terminal_v': {
                'vshape': (T, N, 1)
            }
        }
        self.keys = list(self.scheme.keys())
        self.reset()

    def reset(self):
        '''Allocates space for containers.'''
        for k, info in self.scheme.items():
            assert 'vshape' in info, f'Scheme must define vshape for {k}'
            vshape = info['vshape']
            dtype = info.get('dtype', np.float32)
            init = info.get('init', np.zeros)
            self.__dict__[k] = init(vshape, dtype=dtype)
        self.t = 0

    def push(self, batch):
        '''Inserts transition step data (as dict) to storage.'''
        for k, v in batch.items():
            assert k in self.keys
            shape = self.scheme[k]['vshape'][1:]
            dtype = self.scheme[k].get('dtype', np.float32)
            # print([k, v])
            v_ = np.asarray(deepcopy(v), dtype=dtype).reshape(shape)
            self.__dict__[k][self.t] = v_
        self.t = (self.t + 1) % self.max_length

    def get(self, device='cpu'):
        '''Returns all data.'''
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][2:]
            data = self.__dict__[k].reshape(-1, *shape)
            batch[k] = torch.as_tensor(data, device=device)
        return batch

    def sample(self, indices):
        '''Returns partial data.'''
        batch = {}
        for k, info in self.scheme.items():
            shape = info['vshape'][2:]
            batch[k] = self.__dict__[k].reshape(-1, *shape)[indices]
        return batch

    def sampler(self,
                mini_batch_size,
                device='cpu',
                drop_last=True
                ):
        '''Makes sampler to loop through all data.'''
        total_steps = self.max_length * self.batch_size
        sampler = random_sample(np.arange(total_steps), mini_batch_size, drop_last)
        for indices in sampler:
            batch = self.sample(indices)
            batch = {
                k: torch.as_tensor(v, device=device) for k, v in batch.items()
            }
            yield batch


def random_sample(indices,
                  batch_size,
                  drop_last=True
                  ):
    '''Returns index batches to iterate over.'''
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(
        -1, batch_size)
    for batch in batches:
        yield batch
    if not drop_last:
        r = len(indices) % batch_size
        if r:
            yield indices[-r:]