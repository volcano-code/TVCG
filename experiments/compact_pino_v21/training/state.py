"""Deterministic single-process sampling and safe tensor-only checkpoint state."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch


SCHEDULE_SCHEMA = 'compact_pino.fixed-exposure-schedule.v1'


class ScheduleExhausted(RuntimeError):
    """A frozen exposure schedule never silently cycles or invents more rows."""


def load_exposure_schedule(path, expected_sha256, manifest_sha256, loader_record_ids):
    """Bind ordered *occurrences* to the exact manifest and admitted train IDs."""
    if not isinstance(expected_sha256, str) or re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is None:
        raise ValueError('fixed schedule requires an explicit lowercase SHA256 in TrainConfig')
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('exposure schedule SHA256 mismatch')
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get('schema') != SCHEDULE_SCHEMA:
        raise ValueError('unsupported exposure schedule schema')
    if value.get('training_manifest_sha256') != manifest_sha256:
        raise ValueError('exposure schedule training manifest identity mismatch')
    ordered = value.get('record_ids')
    admitted = list(loader_record_ids)
    if (not admitted or any(not isinstance(rid, str) or not rid for rid in admitted)
            or len(set(admitted)) != len(admitted)):
        raise ValueError('fixed schedule requires distinct admitted training record IDs')
    admitted_set = set(admitted)
    if (not isinstance(ordered, list) or not ordered
            or any(not isinstance(rid, str) or rid not in admitted_set for rid in ordered)):
        raise ValueError('schedule occurrences must reference only admitted training record IDs')
    identity = dict(schema=SCHEDULE_SCHEMA, sha256=expected_sha256,
                    training_manifest_sha256=manifest_sha256,
                    loader_record_ids=admitted, occurrences=len(ordered),
                    unique_records=len(set(ordered)))
    return FixedExposureSchedule(ordered, admitted, identity), dict(identity, path=str(path))


class FixedExposureSchedule:
    """Finite, immutable ID schedule with an exactly restorable occurrence cursor."""
    def __init__(self, ordered_record_ids, loader_record_ids, identity):
        index = {rid: i for i, rid in enumerate(loader_record_ids)}
        self.order = tuple(index[rid] for rid in ordered_record_ids)
        self.identity = copy.deepcopy(identity)
        self.count = len(self.order)
        self.cursor = 0
        self.epoch = 0

    @property
    def remaining(self):
        return self.count - self.cursor

    def next_batch(self, size):
        if type(size) is not int or size < 1:
            raise ValueError('positive batch size required')
        if size > self.remaining:
            raise ScheduleExhausted('fixed exposure schedule exhausted; no partial batch or implicit cycle')
        result = list(self.order[self.cursor:self.cursor+size])
        self.cursor += size
        return result

    def state_dict(self):
        return dict(schema=SCHEDULE_SCHEMA, identity=copy.deepcopy(self.identity),
                    count=self.count, cursor=self.cursor, epoch=0)

    def load_state_dict(self, state):
        if (state.get('schema') != SCHEDULE_SCHEMA or state.get('identity') != self.identity
                or state.get('count') != self.count or state.get('epoch') != 0
                or type(state.get('cursor')) is not int or not 0 <= state['cursor'] <= self.count):
            raise ValueError('invalid or foreign fixed exposure schedule checkpoint')
        self.cursor = state['cursor']


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
