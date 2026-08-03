"""Shared benchmark / autotune / validation harness.

Design rules this file enforces, because they are what make the fused-vs-unfused
comparison meaningful:

1.  **Chain timing.** An unfused variant is a *sequence* of kernels. We time the whole
    sequence as one unit, with a single L2 flush before the sequence -- not between its
    kernels. Flushing between them would fabricate a fusion win, because in real
    execution the producer's output is still resident in L2 when the consumer starts.

2.  **Independent tuning.** `autotune()` searches each variant's own config space and
    returns that variant's own optimum. A fused kernel and its unfused counterpart never
    share a config. Comparing a tuned kernel against an untuned one is the single easiest
    way to manufacture a fake result.

3.  **Same source, flags differ.** Kernels are written once with `tl.constexpr` flags
    selecting the fused epilogue/prologue. The unfused variant runs the same kernel with
    the flag off plus a separate kernel for the split-out work. Mapping (tile sizes,
    warps, stages, loop order) is the only thing allowed to differ.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

# `GLM52_RESULTS_DIR` keeps a port's results from overwriting another platform's. The C500
# baseline lives in results/c500/; the RTX 4060 port writes to results/rtx4060/.
RESULTS_DIR = Path(
    os.environ.get("GLM52_RESULTS_DIR", Path(__file__).resolve().parent.parent / "results")
)

# Buffer used to evict L2 between measurements. It must exceed the device's L2: C500 is
# 8 MB, RTX 4060 (Ada) is 32 MB, so 128 MB clears both with margin. Sized from the device
# rather than assumed, because a flush smaller than L2 silently turns every measurement
# into a warm-cache one -- which flatters the arm that re-reads an intermediate.
_FLUSH_BYTES = max(128 * 2**20, 4 * torch.cuda.get_device_properties(0).L2_cache_size) \
    if torch.cuda.is_available() else 128 * 2**20
_flush_buf: torch.Tensor | None = None


def _flush_l2() -> None:
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(_FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    _flush_buf.zero_()


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------
@dataclass
class Timing:
    p50_ms: float
    p10_ms: float
    p90_ms: float
    mean_ms: float
    n: int
    noflush_p50_ms: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)


def bench_chain(
    fns: Sequence[Callable[[], object]],
    warmup: int = 25,
    rep: int = 100,
    flush: bool = True,
) -> Timing:
    """Time `fns` executed back-to-back as a single logical operation.

    One L2 flush happens before each timed repetition of the whole chain (see rule 1).
    Returns median / p10 / p90 over `rep` repetitions, plus a no-flush median for
    reference (relevant to tiny decode kernels where launch overhead dominates).
    """
    if callable(fns):
        fns = [fns]

    for _ in range(warmup):
        for fn in fns:
            fn()
    torch.cuda.synchronize()

    def _measure(do_flush: bool) -> list[float]:
        start = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        for i in range(rep):
            if do_flush:
                _flush_l2()
            start[i].record()
            for fn in fns:
                fn()
            end[i].record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in zip(start, end)]

    times = sorted(_measure(flush))
    noflush = float("nan")
    try:
        nf = sorted(_measure(False))
        noflush = nf[len(nf) // 2]
    except Exception:
        pass

    n = len(times)
    return Timing(
        p50_ms=times[n // 2],
        p10_ms=times[max(0, int(0.1 * n))],
        p90_ms=times[min(n - 1, int(0.9 * n))],
        mean_ms=statistics.fmean(times),
        n=n,
        noflush_p50_ms=noflush,
    )


# --------------------------------------------------------------------------------------
# Autotuning
# --------------------------------------------------------------------------------------
@dataclass
class TuneResult:
    best_cfg: dict
    best_ms: float
    n_tried: int
    n_failed: int
    table: list = field(default_factory=list)  # [(cfg, ms|None, err|None), ...]

    def as_dict(self) -> dict:
        return {
            "best_cfg": self.best_cfg,
            "best_ms": self.best_ms,
            "n_tried": self.n_tried,
            "n_failed": self.n_failed,
            "table": self.table,
        }


def autotune(
    make_chain: Callable[[dict], Sequence[Callable[[], object]]],
    configs: Iterable[dict],
    warmup: int = 10,
    rep: int = 30,
    verbose: bool = False,
) -> TuneResult:
    """Brute-force search. `make_chain(cfg)` returns the callables to time for that cfg.

    Configs that fail to compile (SMEM overflow, bad tile shape, register limits) are
    recorded as failures rather than aborting the search -- on C500 the SMEM ceiling is
    64 KB, so a good fraction of an NVIDIA-shaped grid legitimately fails here.
    """
    best_ms, best_cfg = float("inf"), None
    table, n_failed = [], 0
    cfgs = list(configs)
    for cfg in cfgs:
        try:
            chain = make_chain(cfg)
            t = bench_chain(chain, warmup=warmup, rep=rep, flush=True)
            table.append((cfg, t.p50_ms, None))
            if t.p50_ms < best_ms:
                best_ms, best_cfg = t.p50_ms, cfg
            if verbose:
                print(f"  {cfg} -> {t.p50_ms:.4f} ms", flush=True)
        except Exception as exc:  # noqa: BLE001 - deliberate: keep searching
            n_failed += 1
            table.append((cfg, None, f"{type(exc).__name__}: {exc}"[:200]))
            if verbose:
                print(f"  {cfg} -> FAIL {type(exc).__name__}", flush=True)
        finally:
            torch.cuda.empty_cache()
    if best_cfg is None:
        raise RuntimeError(f"every one of {len(cfgs)} configs failed")
    return TuneResult(best_cfg, best_ms, len(cfgs), n_failed, table)


# --------------------------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------------------------
def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Max-abs relative error against the reference's dynamic range."""
    got32, ref32 = got.float(), ref.float()
    denom = ref32.abs().max().clamp_min(1e-6)
    return ((got32 - ref32).abs().max() / denom).item()


def check(
    got: torch.Tensor, ref: torch.Tensor, tol: float = 2e-2, label: str = ""
) -> dict:
    """bf16 chains through K=6144..32768 accumulate real error; tol is on the *relative*
    max-abs error, and the reference is computed in fp32."""
    err = rel_err(got, ref)
    ok = err <= tol and torch.isfinite(got).all().item()
    return {"label": label, "rel_err": err, "tol": tol, "ok": bool(ok)}


# --------------------------------------------------------------------------------------
# Result recording
# --------------------------------------------------------------------------------------
def record(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    payload = dict(payload)
    payload.setdefault("_meta", {})
    payload["_meta"].update(
        {
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": torch.cuda.get_device_name(0),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "torch": torch.__version__,
        }
    )
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def speedup_row(
    regime: str, fused: Timing, unfused: Timing, extra: dict | None = None
) -> dict:
    row = {
        "regime": regime,
        "fused_ms": fused.p50_ms,
        "unfused_ms": unfused.p50_ms,
        "speedup": unfused.p50_ms / fused.p50_ms if fused.p50_ms > 0 else float("nan"),
        "fused_p10_p90": [fused.p10_ms, fused.p90_ms],
        "unfused_p10_p90": [unfused.p10_ms, unfused.p90_ms],
    }
    if extra:
        row.update(extra)
    return row


def main_guard(fn: Callable[[], None]) -> None:
    """Run `fn`, printing a full traceback to stdout on failure (workflow agents read
    stdout, and a bare exception message is not enough to debug a Triton compile)."""
    try:
        fn()
    except Exception:
        traceback.print_exc()
        raise
