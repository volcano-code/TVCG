"""Deterministic single-process sampling and safe tensor-only checkpoint state."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch


class StatefulShuffle:
    """Visit every eligible record exactly once per epoch; no prefetch workers."""
    def __init__(self, count, seed):
        if type(count) is not int or count < 1:
            raise ValueError('positive sampler size required')
        self.count = count
        self.generator = np.random.default_rng(seed)
        self.order = self.generator.permutation(count).tolist()
        self.cursor = 0
        self.epoch = 0

    def next_batch(self, size):
        if type(size) is not int or size < 1:
            raise ValueError('positive batch size required')
        result = []
        while len(result) < size:
            if self.cursor == self.count:
                self.order = self.generator.permutation(self.count).tolist()
                self.cursor = 0
                self.epoch += 1
            take = min(size-len(result),self.count-self.cursor)
            result.extend(self.order[self.cursor:self.cursor+take])
            self.cursor += take
        return result

    def state_dict(self):
        return dict(count=self.count,order=list(self.order),cursor=self.cursor,epoch=self.epoch,
                    generator=copy.deepcopy(self.generator.bit_generator.state))

    def load_state_dict(self,state):
        if (state['count'] != self.count or sorted(state['order']) != list(range(self.count))
                or type(state['cursor']) is not int or not 0 <= state['cursor'] <= self.count
                or type(state['epoch']) is not int or state['epoch'] < 0):
            raise ValueError('invalid or foreign sampler checkpoint')
        self.order,self.cursor,self.epoch = list(state['order']),state['cursor'],state['epoch']
        self.generator.bit_generator.state = copy.deepcopy(state['generator'])


def capture_rng():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(),
                numpy=dict(algorithm=numpy_state[0],keys=numpy_state[1].tolist(),
                           position=numpy_state[2],has_gauss=numpy_state[3],cached_gaussian=numpy_state[4]),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    item = state['numpy']
    np.random.set_state((item['algorithm'],np.asarray(item['keys'],dtype=np.uint32),
                         item['position'],item['has_gauss'],item['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('checkpoint CUDA RNG topology differs from runtime')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])


def seed_everything(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False


def observed_optimizer_step(optimizer, scaler):
    """Observe the real post-step hook, not GradScaler.step's None return value."""
    called = []
    handle = optimizer.register_step_post_hook(lambda *args, **kwargs: called.append(True))
    try:
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    finally:
        handle.remove()
    return bool(called)
