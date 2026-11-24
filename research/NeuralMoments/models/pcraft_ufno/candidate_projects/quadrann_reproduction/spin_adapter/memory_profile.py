"""Opt-in, boundary-synchronized memory and timing measurements for one update."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import json
import os
from pathlib import Path
import time
import weakref

import torch


_ACTIVE = ContextVar("unified_operator_memory_profile", default=None)


def capture_profile():
    """Bind graph callbacks to their originating session, without owning the session."""
    profiler = _ACTIVE.get()
    return weakref.ref(profiler) if profiler is not None else None


@contextmanager
def use_profile(handle):
    """Autograd engine worker threads do not inherit Python context variables."""
    profiler = handle() if handle is not None else None
    token = _ACTIVE.set(profiler if profiler is not None and profiler.running else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def profile_stage(name, phase="forward", **counts):
    profiler = _ACTIVE.get()
    return profiler.stage(name, phase, counts) if profiler is not None else nullcontext()


def profile_count(**counts):
    profiler = _ACTIVE.get()
    if profiler is not None:
        profiler.record_counts(counts)


def profile_tensors(**named_tensors):
    profiler = _ACTIVE.get()
    if profiler is not None:
        profiler.record_tensors(named_tensors)


def profile_cache_tensors(cache):
    profiler = _ACTIVE.get()
    if profiler is None:
        return
    children=[("shared",cache.shared_cache)]
    if hasattr(cache,"sqno_cache"):children.append(("sqno",cache.sqno_cache))
    if hasattr(cache,"token_cache"):children.append(("tokens",cache.token_cache))
    named = {f"{prefix}_{name}": value
             for prefix, obj in children
             for name, value in vars(obj).items() if torch.is_tensor(value)}
    profiler.record_tensors(named)
    storage = {}
    for name, value in named.items():
        buffer = value.untyped_storage()
        key = str(value.device), buffer.data_ptr()
        group = storage.setdefault(key, {"device": key[0], "bytes": buffer.nbytes(), "fields": []})
        group["fields"].append(name)
    profiler.stack[-1]["row"]["cache_storage"] = {
        "tensor_fields": len(named), "distinct_storages": len(storage),
        "distinct_storage_bytes": sum(group["bytes"] for group in storage.values()),
        "alias_groups": [group for group in storage.values() if len(group["fields"]) > 1],
        "scope": "tensor-valued fields of geometry and response caches at encoder return",
    }


class _ReplayRegion:
    """Non-reentrant checkpoint can enter the same context on several VJPs."""
    def __init__(self, name):
        self.name = name
        self.handle = capture_profile()
        self.contexts = []

    @contextmanager
    def _activate(self):
        with use_profile(self.handle):
            profiler = _ACTIVE.get()
            with profiler.recompute(self.name) if profiler is not None else nullcontext():
                yield

    def __enter__(self):
        context = self._activate()
        self.contexts.append(context)
        return context.__enter__()

    def __exit__(self, *exc):
        return self.contexts.pop().__exit__(*exc)


def profile_recompute(name):
    """Measure fine-grained replay on the current stream without per-tile sync."""
    return _ReplayRegion(name) if _ACTIVE.get() is not None else nullcontext()


def profile_adjoint(name, outputs, inputs=(), parameters=()):
    """Observe the span from the first output gradient to the last observed input/parameter gradient.

    Hooks do not wrap tensors, keep their values, modify gradients or synchronize CUDA.
    Only callbacks run inside an explicit training Backward stage contribute measurements.
    """
    profiler = _ACTIVE.get()
    if profiler is None or not profiler.stack or profiler.stack[-1]["phase"] != "forward":
        return
    handle = capture_profile()
    spans = {}

    def observe(kind):
        def hook(grad):
            owner = handle()
            if owner is None or not owner.running or not owner.training_backward:
                return
            task = torch._C._current_graph_task_id()
            if kind == "output" and task not in spans:
                sample = {"start_cpu": time.perf_counter(), "output": 0, "input": 0, "parameter": 0,
                          "start_event": None, "end_event": None}
                spans[task] = sample
                owner.adjoint_samples.setdefault(name, []).append(sample)
                if owner.device.type == "cuda":
                    sample["start_event"] = torch.cuda.Event(enable_timing=True)
                    sample["start_event"].record(torch.cuda.current_stream(owner.device))
            sample = spans.get(task)
            if sample is None:
                return
            sample[kind] += 1
            if kind != "output":
                sample["end_cpu"] = time.perf_counter()
                if owner.device.type == "cuda":
                    if sample["end_event"] is None:
                        sample["end_event"] = torch.cuda.Event(enable_timing=True)
                    sample["end_event"].record(torch.cuda.current_stream(owner.device))
        return hook

    profiler.adjoint_samples.setdefault(name, [])
    for kind, tensors in (("output", outputs), ("input", inputs), ("parameter", parameters)):
        seen = set()
        callback = observe(kind)
        for tensor in (tensors,) if torch.is_tensor(tensors) else tensors:
            if torch.is_tensor(tensor) and tensor.requires_grad and id(tensor) not in seen:
                seen.add(id(tensor))
                profiler.hook_handles.append(tensor.register_hook(callback))


def _adjoint_report(samples, cuda):
    result = {}
    for name, rows in samples.items():
        complete = [row for row in rows if "end_cpu" in row]
        result[name] = {
            "status": "measured" if rows and len(complete) == len(rows) else
                      ("partial" if rows else "no_output_gradient_observed"),
            "started_spans": len(rows), "completed_spans": len(complete),
            "output_gradients": sum(row["output"] for row in rows),
            "input_gradients": sum(row["input"] for row in complete),
            "parameter_gradients": sum(row["parameter"] for row in complete),
            "cpu_wall_seconds": sum(row["end_cpu"] - row["start_cpu"] for row in complete) if complete else None,
            "cuda_stream_seconds": (sum(row["start_event"].elapsed_time(row["end_event"]) for row in complete)
                                    / 1000 if complete and cuda else None),
        }
    return result


def _array_storage_bytes(array):
    owner = array
    while getattr(owner, "base", None) is not None and hasattr(owner.base, "nbytes"):
        owner = owner.base
    return int(owner.nbytes)


def _rss():
    with open("/proc/self/statm", encoding="ascii") as stream:
        return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class MemoryProfile:
    """Reset and merge allocator high-water marks at every stage boundary.

    Stage times are inclusive. Phase times sum exclusive wall-clock segments.
    Only integers and strings are retained, never tensors or autograd graphs.
    """

    def __init__(self, device, output):
        self.device = torch.device(device)
        self.output = Path(output)
        self.stack = []
        self.stages = {}
        self.metadata = {}
        self.phase_seconds = Counter()
        self.replay_events = []
        self.replay_times = {}
        self.running = False
        self.training_backward = False
        self.adjoint_samples = {}
        self.hook_handles = []

    def _sample(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            stats = torch.cuda.memory_stats(self.device)
            allocated = stats.get("allocated_bytes.all.current", 0)
            reserved = stats.get("reserved_bytes.all.current", 0)
            peak_allocated = stats.get("allocated_bytes.all.peak", allocated)
            peak_reserved = stats.get("reserved_bytes.all.peak", reserved)
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            allocated = reserved = peak_allocated = peak_reserved = 0
        return allocated, reserved, peak_allocated, peak_reserved, _rss(), time.perf_counter()

    def _boundary(self):
        allocated, reserved, peak_a, peak_r, rss, now = self._sample()
        self.peak_allocated = max(self.peak_allocated, peak_a)
        self.peak_reserved = max(self.peak_reserved, peak_r)
        self.peak_rss = max(self.peak_rss, rss)
        for frame in self.stack:
            row = frame["row"]
            row["peak_allocated_bytes"] = max(row["peak_allocated_bytes"], peak_a)
            row["peak_reserved_bytes"] = max(row["peak_reserved_bytes"], peak_r)
            row["rss_max_boundary_bytes"] = max(row["rss_max_boundary_bytes"], rss)
        phase = self.stack[-1]["phase"] if self.stack else "other"
        elapsed = now - self.previous
        self.phase_seconds[phase] += elapsed
        if self.stack:
            self.stack[-1]["row"]["exclusive_seconds"] += elapsed
        self.previous = now
        return allocated, reserved, rss, now

    @contextmanager
    def stage(self, name, phase, counts):
        a, r, rss, start = self._boundary()
        if self.stack and self.stack[-1]["phase"] == "verification":
            phase = "verification"
        elif self.stack and phase == "forward" and self.stack[-1]["phase"] in ("backward", "recompute"):
            phase = self.stack[-1]["phase"]
        row = self.stages.setdefault((name, phase), {
            "name": name, "phase": phase, "calls": 0,
            "allocated_before_bytes": a, "reserved_before_bytes": r,
            "peak_allocated_bytes": a, "peak_reserved_bytes": r,
            "rss_first_bytes": rss, "rss_max_boundary_bytes": rss,
            "inclusive_seconds": 0., "exclusive_seconds": 0.,
            "counts": Counter(), "tensor_observations": {}, "observed_tensor_count": 0,
            "observed_elements": 0,
        })
        row["calls"] += 1
        self.stack.append({"phase": phase, "row": row})
        old_backward = self.training_backward
        if phase == "backward" and name == "Backward":
            self.training_backward = True
        try:
            self.record_counts(counts)
            yield
        finally:
            try:
                a, r, rss, end = self._boundary()
                row.update(allocated_after_bytes=a, reserved_after_bytes=r, rss_last_bytes=rss)
                row["inclusive_seconds"] += end - start
            finally:
                self.training_backward = old_backward
                self.stack.pop()

    @contextmanager
    def recompute(self, name):
        start = time.perf_counter()
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(stream)
        try:
            yield
        finally:
            timings = self.replay_times.setdefault(name, {
                "calls": 0, "cpu_wall_seconds": 0.,
                "cuda_stream_seconds": 0. if self.device.type == "cuda" else None,
            })
            timings["calls"] += 1
            timings["cpu_wall_seconds"] += time.perf_counter() - start
            if self.device.type == "cuda":
                end.record(stream)
                self.replay_events.append((name, begin, end))

    def record_counts(self, counts):
        if not self.stack:
            raise RuntimeError("profile_count requires a profile_stage")
        self.stack[-1]["row"]["counts"].update({k: int(v) for k, v in counts.items()})

    def record_tensors(self, named_tensors):
        if not self.stack:
            raise RuntimeError("profile_tensors requires a profile_stage")
        row = self.stack[-1]["row"]
        observations = row["tensor_observations"]
        for name, tensor in named_tensors.items():
            is_tensor = torch.is_tensor(tensor)
            elements = tensor.numel() if is_tensor else int(tensor.size)
            if name not in observations:
                grad_fn = tensor.grad_fn if is_tensor else None
                observations[name] = {
                    "count": 0, "total_elements": 0, "max_elements": 0,
                    "first_shape": list(tensor.shape), "first_dtype": str(tensor.dtype),
                    "first_device": str(tensor.device) if is_tensor else "cpu",
                    "first_requires_grad": bool(is_tensor and tensor.requires_grad),
                    "first_grad_fn": type(grad_fn).__name__ if grad_fn is not None else None,
                    "first_storage_bytes": (tensor.untyped_storage().nbytes()
                                            if is_tensor else _array_storage_bytes(tensor)),
                }
            row["observed_tensor_count"] += 1
            row["observed_elements"] += elements
            observation = observations[name]
            observation["count"] += 1
            observation["total_elements"] += elements
            observation["max_elements"] = max(observation["max_elements"], elements)

    def __enter__(self):
        if _ACTIVE.get() is not None:
            raise RuntimeError("nested profile sessions would reset the same allocator counters")
        a, r, _, _, rss, now = self._sample()
        self.baseline = {"allocated_bytes": a, "reserved_bytes": r, "rss_bytes": rss}
        self.peak_allocated, self.peak_reserved, self.peak_rss = a, r, rss
        self.previous = self.started = now
        self.running = True
        self.token = _ACTIVE.set(self)
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._boundary()
            for name, begin, end in self.replay_events:
                self.replay_times[name]["cuda_stream_seconds"] += begin.elapsed_time(end) / 1000
            self.replay_events.clear()
            data = {
                "schema": "tvcg.unified-memory-profile.v1",
                "status": "failed" if exc_type else "complete",
                "device": str(self.device), "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
                "cuda_measurements_available": self.device.type == "cuda",
                "baseline": self.baseline,
                "peak_allocated_bytes": self.peak_allocated,
                "peak_reserved_bytes": self.peak_reserved,
                "rss_max_boundary_bytes": self.peak_rss,
                "elapsed_seconds": self.previous - self.started,
                "exclusive_phase_seconds": dict(self.phase_seconds),
                "recompute_regions": self.replay_times,
                "adjoint_spans": _adjoint_report(self.adjoint_samples, self.device.type == "cuda"),
                "stages": list(self.stages.values()), "metadata": self.metadata,
                "measurement": (
                    "CUDA allocator peaks merge adjacent synchronized segments, including nested scopes. "
                    "Stage times are inclusive, phase times exclusive. RSS is current process RSS sampled "
                    "at boundaries. Repeated stages aggregate counts, durations, first entry, last exit and maxima. "
                    "Named tensor observations count logical elements each time objects are observed. "
                    "Fine-grained recompute regions measure host wall time and current-stream CUDA event time "
                    "inside their parent phase; their times may overlap and must not be added to wall time. "
                    "A missing timing region is unmeasured. The Backward stage covers the entire autograd call. "
                    "Named replay regions only cover recomputation, not the surrounding gradient kernels. "
                    "Adjoint spans run from first observed output gradient to last observed input or parameter "
                    "gradient inside the training backward. Shared inputs/parameters can extend and overlap spans. "
                    "These are inclusive gradient-arrival spans and must not be summed. I/O, optimizer update, "
                    "and immutable metadata have no parameter backward."
                ),
            }
            if exc_type:
                data["error"] = f"{exc_type.__name__}: {exc}"
            self.write(data)
        finally:
            self.running = False
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles.clear()
            _ACTIVE.reset(self.token)

    def write(self, data):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        lines = [
            "# Unified operator memory profile", "",
            f"Device `{self.device}`. Status `{data['status']}`. Wall time {data['elapsed_seconds']:.3f} s.",
            f"Peak allocated {self.peak_allocated / 2**30:.3f} GiB. Peak reserved "
            f"{self.peak_reserved / 2**30:.3f} GiB. Maximum sampled RSS {self.peak_rss / 2**30:.3f} GiB.",
            "", data["measurement"], "", "| Phase | Exclusive seconds |", "|---|---:|",
        ]
        for phase, duration in data["exclusive_phase_seconds"].items():
            lines.append(f"| {phase} | {duration:.4f} |")
        if self.replay_times:
            lines.extend(["", "| Recompute region | Calls | CPU wall s | CUDA stream s |", "|---|---:|---:|---:|"])
            for name, timings in self.replay_times.items():
                gpu = timings["cuda_stream_seconds"]
                gpu_text = f"{gpu:.4f}" if gpu is not None else "—"
                lines.append(f"| {name} | {timings['calls']} | {timings['cpu_wall_seconds']:.4f} | {gpu_text} |")
        if data["adjoint_spans"]:
            lines.extend(["", "| Training adjoint span | Completed / started | Parameter gradients | CPU wall s | CUDA span s |",
                          "|---|---:|---:|---:|---:|"])
            for name, row in data["adjoint_spans"].items():
                cpu, gpu = row["cpu_wall_seconds"], row["cuda_stream_seconds"]
                cpu_text = f"{cpu:.4f}" if cpu is not None else "—"
                gpu_text = f"{gpu:.4f}" if gpu is not None else "—"
                lines.append(f"| {name} | {row['completed_spans']} / {row['started_spans']} | "
                             f"{row['parameter_gradients']} | {cpu_text} | {gpu_text} |")
        lines.extend(["", "| Stage | Phase | Calls | Inclusive s | Peak allocated GiB | Peak reserved GiB | Sampled RSS GiB |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for row in data["stages"]:
            lines.append(f"| {row['name']} | {row['phase']} | {row['calls']} | {row['inclusive_seconds']:.4f} | "
                         f"{row['peak_allocated_bytes']/2**30:.4f} | {row['peak_reserved_bytes']/2**30:.4f} | "
                         f"{row['rss_max_boundary_bytes']/2**30:.4f} |")
        if self.metadata:
            lines.extend(["", "## Batch and update", "", "```json", json.dumps(self.metadata, indent=2), "```"])
        if "error" in data:
            lines.extend(["", data["error"]])
        self.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
