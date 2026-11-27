"""Bounded, reusable execution of independent finite-surface point queries.

Only scheduling lives here.  The closest-feature kernel, point order and all
geometric tolerances stay in v21_response_surface.  A context owns at most one
lazy thread pool; repeated chunks and different surfaces can share that pool.

Use ``with SurfaceExecutionContext(...)`` around a solve for an explicit
lifetime, or pass a context to ``prepare_surface``.  The compatibility default
is one lazy process pool, released by ``shutdown_default_surface_execution``
or process exit.  No geometry, witnesses, motion or model tensors are cached.
"""
from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import os
from threading import RLock

import numpy as np


def _cpu_budget():
    """Return the caller's available CPU count and an affinity identity.

    Linux worker threads inherit the creating thread's affinity.  Retiring the
    pool when the *set*, not just its size, changes avoids keeping workers on
    cores no longer allowed for the caller.  Hosts without affinity support use
    the process-aware count when available, then the system count.
    """
    affinity = getattr(os, 'sched_getaffinity', None)
    if affinity is not None:
        try:
            cpus = tuple(sorted(affinity(0)))
        except (OSError, NotImplementedError):
            cpus = ()
        if cpus:
            return len(cpus), ('affinity', cpus)
    process_count = getattr(os, 'process_cpu_count', None)
    count = process_count() if process_count is not None else None
    count = count or os.cpu_count() or 1
    count = max(1, int(count))
    return count, ('cpu_count', count)


_active_execution: ContextVar = ContextVar('compact_pino_surface_execution', default=None)


class SurfaceExecutionContext:
    """Own the resources for finite-surface queries, with ordered results.

    ``max_workers=None`` uses the historical ceiling of eight, further capped
    by CPU affinity.  ``max_workers=1`` is the serial reference path.  The default
    threshold of 8192 points per partition is retained.  Queries sharing one
    context are scheduled serially; the independent partitions within each
    query may run in parallel.  This prevents simultaneous callers from each
    creating another full pool.  Close waits for active work, and reuse of a
    closed or fork-inherited context is an error, never a silent fallback.
    """

    def __init__(self, max_workers=None, *, points_per_worker=8192):
        max_workers = 8 if max_workers is None else max_workers
        for name, value in (('max_workers', max_workers), ('points_per_worker', points_per_worker)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.max_workers = max_workers
        self.points_per_worker = points_per_worker
        self._pid = os.getpid()
        self._lock = RLock()
        self._pool = None
        self._pool_policy = None
        self._closed = False
        self._activation = None
        self._counts = dict(queries=0, serial_queries=0, parallel_queries=0,
                            pool_creations=0, pool_shutdowns=0, failed_queries=0,
                            submitted_partitions=0)
        self._last_workers = 0
        self._last_available_cpus = None
        self._last_affinity = None

    def _check_process(self):
        # Check before acquiring a potentially inherited lock.
        if os.getpid() != self._pid:
            raise RuntimeError('surface execution context belongs to another process; create one after fork')

    def _check_open(self):
        if self._closed:
            raise RuntimeError('surface execution context is closed')

    def _retire_pool(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
            self._pool_policy = None
            self._counts['pool_shutdowns'] += 1

    def query(self, kernel, points, *arguments):
        """Run the unchanged per-point kernel and concatenate in input order."""
        self._check_process()
        with self._lock:
            self._check_open()
            available, affinity = _cpu_budget()
            capacity = min(self.max_workers, available)
            workers = min(capacity, max(1, len(points) // self.points_per_worker))
            policy = (capacity, affinity)
            if self._pool is not None and self._pool_policy != policy:
                self._retire_pool()
            self._last_workers = workers
            self._last_available_cpus = available
            self._last_affinity = affinity
            self._counts['queries'] += 1
            if workers == 1:
                self._counts['serial_queries'] += 1
                try:
                    return kernel(points, *arguments)
                except BaseException:
                    self._counts['failed_queries'] += 1
                    raise
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=capacity,
                                                thread_name_prefix='cpino-surface')
                self._pool_policy = policy
                self._counts['pool_creations'] += 1
            self._counts['parallel_queries'] += 1
            futures = []
            try:
                for part in np.array_split(points, workers):
                    futures.append(self._pool.submit(kernel, part, *arguments))
                    self._counts['submitted_partitions'] += 1
                # Future completion order must not change witness/node order.
                outputs = [future.result() for future in futures]
                return tuple(np.concatenate(parts) for parts in zip(*outputs))
            except BaseException:
                self._counts['failed_queries'] += 1
                # Match the old per-query pool's lifetime on failure: no worker
                # may retain this query's arrays after the exception is returned.
                for future in futures:
                    future.cancel()
                for future in futures:
                    try:
                        future.result()
                    except BaseException:
                        pass
                raise

    def snapshot(self):
        self._check_process()
        with self._lock:
            return dict(self._counts, max_workers=self.max_workers,
                        points_per_worker=self.points_per_worker,
                        last_workers=self._last_workers,
                        last_available_cpus=self._last_available_cpus,
                        last_affinity=self._last_affinity,
                        pool_capacity=0 if self._pool_policy is None else self._pool_policy[0],
                        closed=self._closed, owner_pid=self._pid)

    def close(self):
        """Wait for queries, release workers; repeated close is harmless."""
        self._check_process()
        with self._lock:
            if not self._closed:
                self._retire_pool()
                self._closed = True

    def __enter__(self):
        self._check_process()
        with self._lock:
            self._check_open()
            if self._activation is not None:
                raise RuntimeError('surface execution context is already active')
            self._activation = _active_execution.set(self)
        return self

    def __exit__(self, *exception):
        # ContextVar enforces restoring in the same thread/context as entry.
        token = self._activation
        if token is None:
            raise RuntimeError('surface execution context is not active')
        _active_execution.reset(token)
        self._activation = None
        self.close()
        return False


_default_lock = RLock()
_default_execution = None


def get_surface_execution_context():
    """Use an explicit current scope, otherwise the lazy process default."""
    active = _active_execution.get()
    if active is not None:
        return active
    global _default_execution
    with _default_lock:
        if _default_execution is None:
            _default_execution = SurfaceExecutionContext()
        return _default_execution


def shutdown_default_surface_execution():
    """Release the compatibility default; the next unscoped call starts fresh.

    Call after outstanding solves finish.  Explicit contexts have their own
    lifetime and are not closed by this function.
    """
    global _default_execution
    with _default_lock:
        previous, _default_execution = _default_execution, None
    if previous is not None:
        previous.close()


def _after_fork():
    # Never join or acquire the parent process's threads/locks in a child.
    # An explicitly inherited active scope still rejects use via _check_process.
    global _default_lock, _default_execution
    _default_lock = RLock()
    _default_execution = None


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)
atexit.register(shutdown_default_surface_execution)
