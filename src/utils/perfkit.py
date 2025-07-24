#!/usr/bin/env python3
"""
perfkit.py  · v4
────────────────────────────────────────────────────────
Infrastructure glue for wide / tall / small pipelines

  ⏱  @perfclass      – wall-clock, RSS mem, peak-mem, $ cost
  ⚙️  ParallelMixin   – auto n_jobs, threads↔processes heuristic
  ⚡  GPUMixin        – lazy CuPy cache, ks_fast, rand_choice
  🛠  PerfMixin       – one-stop bundle (inherits the 3 above)

NEW vs v3 ──────────────────────────────────────────────────────
  1. Heuristic   prefer="threads" ↔ "processes"  (idea #1)
  2. Lazy, cached CuPy import                      (#2)
  3. Global PEAK_RSS tracker  +  CSV exporter      (#3, #4)
  4. FAST_MODE env flag skips heavy profiling      (#17)
  5. MAX_RAM_FRACTION guard                        (#18)
"""
from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import pickle
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import psutil
from joblib import Parallel, cpu_count, delayed

log = logging.getLogger("perfkit")

# ─────────── module-level high-water mark (idea #4) ────────────
_PEAK_RSS_MB: float = 0.0
_proc = psutil.Process(os.getpid())


# ═══════════════════════════════════════════════════════════════
# 1 ▸  CLASS-LEVEL PROFILER  (@perfclass)       ideas #3 #4 #17 #18
# ═══════════════════════════════════════════════════════════════
def perfclass(
    price_per_min: float = 0.0,
    skip: Callable[[str], bool] = lambda m: m.startswith("_"),
):
    """
    Decorate a *class* so every **public** method is profiled.

    Adds
        · self._perf_log : List[Dict]
        · self.report()  : aggregated list
        · self.export_csv(path)
    """

    def decorate(cls):

        orig_init = cls.__init__

        @functools.wraps(orig_init)
        def __init__(self, *a, **kw):
            self._perf_log = []
            self._perf_price = price_per_min
            orig_init(self, *a, **kw)

        # ------------- wrapper applied to every method -------------
        def _wrap(name, fn):

            @functools.wraps(fn)
            def inner(self, *a, **kw):
                # FAST_MODE → skip heavy book-keeping
                fast = os.getenv("FAST_MODE", "0") in {"1", "true", "yes"}

                # MAX_RAM_FRACTION guard (idea #18)
                thr = float(os.getenv("MAX_RAM_FRACTION", "95"))
                mem_pct = psutil.virtual_memory().percent
                if mem_pct > thr:
                    raise MemoryError(
                        f"RAM usage {mem_pct:.1f}% exceeded threshold {thr}%"
                    )

                rss_before = _proc.memory_info().rss
                t0 = time.perf_counter()

                out = fn(self, *a, **kw)

                dt = time.perf_counter() - t0
                rss_after = _proc.memory_info().rss

                # update global peak
                global _PEAK_RSS_MB
                _PEAK_RSS_MB = max(_PEAK_RSS_MB, rss_after / 2**20)

                if not fast:
                    self._perf_log.append(
                        dict(
                            method=name,
                            seconds=dt,
                            delta_mb=round((rss_after - rss_before) / 2**20, 3),
                            rss_mb=round(rss_after / 2**20, 1),
                        )
                    )
                return out

            return inner

        # attach wrappers
        for n, m in cls.__dict__.items():
            if inspect.isfunction(m) and not skip(n):
                setattr(cls, n, _wrap(n, m))

        cls.__init__ = __init__

        # ---------- aggregated report & exporters (#3) -------------
        def report(self) -> List[Dict[str, Any]]:
            agg: Dict[str, Dict] = {}
            for rec in self._perf_log:
                d = agg.setdefault(
                    rec["method"], dict(calls=0, seconds=0.0, delta_mb=0.0, rss_mb=0.0)
                )
                d["calls"] += 1
                d["seconds"] += rec["seconds"]
                d["delta_mb"] += rec["delta_mb"]
                d["rss_mb"] = max(d["rss_mb"], rec["rss_mb"])

            return sorted(
                [
                    dict(
                        method=k,
                        calls=v["calls"],
                        seconds=round(v["seconds"], 3),
                        mem_peak_mb=round(v["rss_mb"], 1),
                        mem_Δ_mb=round(v["delta_mb"], 3),
                        cost=round(v["seconds"] / 60 * price_per_min, 4),
                    )
                    for k, v in agg.items()
                ],
                key=lambda r: r["seconds"],
                reverse=True,
            )

        def export_csv(self, path: str = "perf_log.csv"):
            """Save aggregated log to CSV (idea #3)."""
            import csv

            rows = self.report()
            if not rows:
                return
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            log.info("Perf CSV written → %s", path)

        def export_json(self, path: str = "perf_log.json"):
            with open(path, "w") as f:
                json.dump(self.report(), f, indent=2)
            log.info("Perf JSON written → %s", path)

        cls.report = report
        cls.export_csv = export_csv
        cls.export_json = export_json
        return cls

    return decorate


# ═══════════════════════════════════════════════════════════════
# 2 ▸  PARALLEL MIX-IN             idea #1 + self-tuning threshold
# ═══════════════════════════════════════════════════════════════
class ParallelMixin:
    """
    self.parallel_map(fn, items, *, min_tasks=8, prefer="auto")

    prefer="auto"  → threads if tall, processes if *wide*
    Heuristic:   if len(items) > 1000 or os.getenv("WIDE_DATA") → processes.
    """

    def __init__(self, *a, n_jobs: Union[int, float, None] = -1, **kw):
        super().__init__(*a, **kw)  # cooperative

        env = os.getenv("PERF_N_JOBS")
        if n_jobs is None and env is not None:
            try:
                n_jobs = float(env) if "." in env else int(env)
            except ValueError:
                n_jobs = None

        if n_jobs is None:
            n_jobs = 0.5  # default 50 % of cores
        if isinstance(n_jobs, float):
            n_jobs = max(int(cpu_count() * n_jobs), 1)
        self._n_jobs = max(int(n_jobs), 1)

    # -----------------------------------------------------------------
    def _auto_prefer(self, items: Sequence[Any]) -> str:
        if os.getenv("WIDE_DATA"):
            return "processes"
        if len(items) > 1000:  # crude but effective
            return "processes"
        return "threads"

    def parallel_map(
        self,
        fn: Callable[[Any], Any],
        items: Sequence[Any],
        *,
        min_tasks: int = 8,
        prefer: str = "auto",
    ) -> List[Any]:

        if len(items) == 0:
            return []
        if self._n_jobs == 1 or len(items) < min_tasks:
            return [fn(x) for x in items]

        if prefer == "auto":
            prefer = self._auto_prefer(items)

        log.debug(
            "Parallel → %d jobs (%s) × %d tasks", self._n_jobs, prefer, len(items)
        )
        return Parallel(n_jobs=self._n_jobs, prefer=prefer)(
            delayed(fn)(x) for x in items
        )


# ═══════════════════════════════════════════════════════════════
# 3 ▸  GPU MIX-IN                         idea #2 (lazy CuPy cache)
# ═══════════════════════════════════════════════════════════════
class _CuPyLoader:
    """Singleton-style loader to avoid repeated heavy imports.

    On macOS/Apple Silicon → use PyTorch MPS if available,
    else fallback to NumPy (CPU).
    """

    _cached: Optional[Any] = None

    @classmethod
    def load(cls):
        if cls._cached is not None:
            return cls._cached

        # 1️⃣ Try PyTorch + MPS (Apple Silicon)
        try:
            import torch

            if torch.backends.mps.is_available():

                class TorchArrayWrapper:
                    def array(self, data):
                        return torch.tensor(data, device="mps")

                    def to_numpy(self, t):
                        return t.cpu().numpy()

                cls._cached = TorchArrayWrapper()
                log.info(
                    "PyTorch MPS detected – GPU fast-path enabled on macOS/Apple Silicon."
                )
                return cls._cached
        except Exception:
            pass

        # 2️⃣ Fallback to NumPy (CPU)
        cls._cached = np
        log.info("GPU not available – using NumPy (CPU path).")
        return cls._cached


class GPUMixin:
    """
    Adds
        · self.use_gpu
        · self.cuda
        · ks_fast(a,b)
        · rand_choice(arr,size,seed)
    """

    def __init__(self, *a, use_gpu: Optional[bool] = True, **kw):
        super().__init__(*a, **kw)

        # Always "enable" GPU mode
        if use_gpu is None:
            env = os.getenv("PERF_USE_GPU")
            use_gpu = env and env.lower() in {"1", "true", "yes"}

        cp = _CuPyLoader.load() if use_gpu else None
        self.use_gpu = cp is not None
        self.cuda = cp

    # ---- helpers ----------------------------------------------------
    def ks_fast(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.use_gpu:
            return float(
                self.cuda.stats.ks_2samp(
                    self.cuda.asarray(a), self.cuda.asarray(b)
                ).pvalue
            )
        from scipy.stats import ks_2samp

        return float(ks_2samp(a, b).pvalue)

    def rand_choice(self, arr: np.ndarray, size: int, seed: int = 0) -> np.ndarray:
        if self.use_gpu:
            rs = self.cuda.random.default_rng(seed)
            idx = rs.integers(0, len(arr), size=size)
            return self.cuda.asnumpy(self.cuda.asarray(arr)[idx])
        return np.random.default_rng(seed).choice(arr, size=size, replace=True)


# ═══════════════════════════════════════════════════════════════
# 4 ▸  ONE-STOP MIX-IN
# ═══════════════════════════════════════════════════════════════
@perfclass()  # you can raise price_per_min here if you want CPU cost $
class PerfMixin(ParallelMixin, GPUMixin):
    """
    Inherit **one** mix-in to get:

        ⏱  automatic timing + RSS + peak + CSV export
        ⚙️  self.parallel_map(...)   (auto threads/processes)
        ⚡  self.ks_fast, self.rand_choice     (GPU if available)

    Cooperative init keeps kwargs clean.
    """

    def __init__(
        self,
        *a,
        n_jobs: Union[int, float, None] = None,
        use_gpu: Optional[bool] = None,
        **kw,
    ):
        ParallelMixin.__init__(self, *a, n_jobs=n_jobs, **kw)
        GPUMixin.__init__(self, *a, use_gpu=use_gpu, **kw)
        # (no further init – we chained manually)
