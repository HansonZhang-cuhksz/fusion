"""Hardware performance-counter collection on MetaX C500, via MCPTI through ctypes.

MACA ships `libmcpti.so`, a faithful clone of the legacy CUPTI 1.x Event/Metric API.  The
vendor's own `mcProfiler` front-end does not work on this machine (its `perf_exec` writes
empty result databases), and the shipped headers contain **no kernel-launch callback id**
(`mcpti_compute_cbid.h` is a generated stub) so the usual callback-driven collection pattern
is unavailable.  What *is* available is manual event-group control, which needs no C at all:

    set collection mode KERNEL -> enable an event-group set -> launch -> read -> disable

repeated once per pass, then `mcptiMetricGetValue` to turn raw events into a metric.  A
metric needing more events than fit in hardware at once is split by MCPTI into several
"sets"; the kernel is replayed once per set, which is why `launch_fn` must be replayable and
deterministic.

Usage:

    from glm52.mcpti import MCPTI
    with MCPTI() as m:
        vals = m.collect(["dram_read_bytes", "achieved_occupancy"], launch_fn)
"""
from __future__ import annotations

import ctypes
import time
from typing import Callable, Sequence

LIBMCPTI = "/opt/maca/lib/libmcpti.so"
LIBMCRT = "/opt/maca/lib/libmcruntime.so"

_KERNEL_MODE = 1                     # MCPTI_EVENT_COLLECTION_MODE_KERNEL
_ATTR_PROFILE_ALL = 1                # MCPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES
_ATTR_INSTANCE_COUNT = 5             # MCPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT
_ATTR_EVENT_DOMAIN_ID = 0            # MCPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID
_DOMAIN_ATTR_TOTAL_INSTANCE = 3      # MCPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT
_READ_FLAG_NONE = 0
_KIND = {0: "double", 1: "uint64", 2: "percent", 3: "throughput", 4: "int64",
         5: "utilization_level"}


class _EventGroupSet(ctypes.Structure):
    _fields_ = [("numEventGroups", ctypes.c_uint32),
                ("eventGroups", ctypes.POINTER(ctypes.c_void_p))]


class _EventGroupSets(ctypes.Structure):
    _fields_ = [("numSets", ctypes.c_uint32),
                ("sets", ctypes.POINTER(_EventGroupSet))]


class _MetricValue(ctypes.Union):
    _fields_ = [("d", ctypes.c_double), ("u64", ctypes.c_uint64),
                ("i64", ctypes.c_int64), ("lvl", ctypes.c_int)]


class MCPTIError(RuntimeError):
    pass


class MCPTI:
    def __init__(self, device: int = 0):
        self.p = ctypes.CDLL(LIBMCPTI, mode=ctypes.RTLD_GLOBAL)
        self.rt = ctypes.CDLL(LIBMCRT, mode=ctypes.RTLD_GLOBAL)
        self.dev = ctypes.c_int(device)
        ctx = ctypes.c_void_p()
        self._ck(self.rt.mcCtxGetCurrent(ctypes.byref(ctx)), "mcCtxGetCurrent")
        if not ctx.value:
            raise MCPTIError("no current MACA context -- initialise torch.cuda first")
        self.ctx = ctx

    # -- plumbing ----------------------------------------------------------------------
    def _ck(self, rc: int, what: str) -> None:
        if rc != 0:
            s = ctypes.c_char_p()
            try:
                self.p.mcptiGetResultString(ctypes.c_int(rc), ctypes.byref(s))
                msg = s.value.decode() if s.value else "?"
            except Exception:
                msg = "?"
            raise MCPTIError(f"{what} failed rc={rc} ({msg})")

    def __enter__(self) -> "MCPTI":
        return self

    def __exit__(self, *a) -> None:
        pass

    def metric_id(self, name: str) -> int:
        mid = ctypes.c_uint32()
        self._ck(self.p.mcptiMetricGetIdFromName(
            self.dev, ctypes.c_char_p(name.encode()), ctypes.byref(mid)), f"metric {name}")
        return mid.value

    def metric_kind(self, mid: int) -> str:
        kind, sz = ctypes.c_int(), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
        # MCPTI_METRIC_ATTR_VALUE_KIND == 2
        self._ck(self.p.mcptiMetricGetAttribute(ctypes.c_uint32(mid), ctypes.c_int(2),
                                                ctypes.byref(sz), ctypes.byref(kind)),
                 "metricGetAttribute")
        return _KIND.get(kind.value, f"kind{kind.value}")

    # -- collection --------------------------------------------------------------------
    def collect(self, names: Sequence[str], launch_fn: Callable[[], None],
                warmup: int = 3) -> dict:
        """Collect each metric in `names` for the kernel(s) launched by `launch_fn`.

        `launch_fn` is replayed once per hardware pass, so it must be deterministic and
        must launch exactly the same work every time.
        """
        import torch
        for _ in range(warmup):
            launch_fn()
        torch.cuda.synchronize()

        out = {}
        for name in names:
            try:
                out[name] = self._one(name, launch_fn)
            except MCPTIError as e:
                out[name] = {"error": str(e)}
        return out

    def _one(self, name: str, launch_fn) -> dict:
        import torch
        mid = self.metric_id(name)
        kind = self.metric_kind(mid)

        n_ev = ctypes.c_uint32()
        self._ck(self.p.mcptiMetricGetNumEvents(ctypes.c_uint32(mid), ctypes.byref(n_ev)),
                 "metricGetNumEvents")
        ev_ids = (ctypes.c_uint32 * n_ev.value)()
        sz = ctypes.c_size_t(ctypes.sizeof(ev_ids))
        self._ck(self.p.mcptiMetricEnumEvents(ctypes.c_uint32(mid), ctypes.byref(sz), ev_ids),
                 "metricEnumEvents")

        sets_p = ctypes.POINTER(_EventGroupSets)()
        mid_c = ctypes.c_uint32(mid)
        self._ck(self.p.mcptiMetricCreateEventGroupSets(
            self.ctx, ctypes.c_size_t(4), ctypes.byref(mid_c), ctypes.byref(sets_p)),
            "metricCreateEventGroupSets")
        sets = sets_p.contents

        acc: dict[int, int] = {}
        t_ns = 0
        for si in range(sets.numSets):
            s = sets.sets[si]
            for gi in range(s.numEventGroups):
                g = ctypes.c_void_p(s.eventGroups[gi])
                one = ctypes.c_uint32(1)
                self.p.mcptiEventGroupSetAttribute(
                    g, ctypes.c_int(_ATTR_PROFILE_ALL),
                    ctypes.c_size_t(4), ctypes.byref(one))

            self._ck(self.p.mcptiSetEventCollectionMode(self.ctx, ctypes.c_int(_KERNEL_MODE)),
                     "setEventCollectionMode")
            self._ck(self.p.mcptiEventGroupSetEnable(ctypes.byref(s)), "eventGroupSetEnable")

            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            launch_fn()
            torch.cuda.synchronize()
            t_ns = max(t_ns, time.perf_counter_ns() - t0)

            for gi in range(s.numEventGroups):
                g = ctypes.c_void_p(s.eventGroups[gi])
                inst, isz = ctypes.c_uint32(), ctypes.c_size_t(4)
                self.p.mcptiEventGroupGetAttribute(g, ctypes.c_int(_ATTR_INSTANCE_COUNT),
                                                   ctypes.byref(isz), ctypes.byref(inst))
                n_g, nsz = ctypes.c_uint32(), ctypes.c_size_t(4)
                self.p.mcptiEventGroupGetAttribute(g, ctypes.c_int(3),   # NUM_EVENTS
                                                   ctypes.byref(nsz), ctypes.byref(n_g))
                ninst = max(1, inst.value)
                nev = max(1, n_g.value)
                vals = (ctypes.c_uint64 * (nev * ninst))()
                ids = (ctypes.c_uint32 * nev)()
                vsz = ctypes.c_size_t(ctypes.sizeof(vals))
                isz2 = ctypes.c_size_t(ctypes.sizeof(ids))
                nread = ctypes.c_size_t()
                rc = self.p.mcptiEventGroupReadAllEvents(
                    g, ctypes.c_int(_READ_FLAG_NONE), ctypes.byref(vsz), vals,
                    ctypes.byref(isz2), ids, ctypes.byref(nread))
                if rc != 0:
                    continue
                got = isz2.value // 4 if isz2.value else nread.value
                for k in range(got):
                    tot = sum(vals[k * ninst + j] for j in range(ninst))
                    acc[ids[k]] = acc.get(ids[k], 0) + tot

            self.p.mcptiEventGroupSetDisable(ctypes.byref(s))

        if not acc:
            raise MCPTIError("no events read")

        id_arr = (ctypes.c_uint32 * len(acc))(*acc.keys())
        val_arr = (ctypes.c_uint64 * len(acc))(*acc.values())
        mv = _MetricValue()
        self._ck(self.p.mcptiMetricGetValue(
            self.dev, ctypes.c_uint32(mid),
            ctypes.c_size_t(ctypes.sizeof(id_arr)), id_arr,
            ctypes.c_size_t(ctypes.sizeof(val_arr)), val_arr,
            ctypes.c_uint64(t_ns), ctypes.byref(mv)), "metricGetValue")

        val = {"double": mv.d, "percent": mv.d, "throughput": mv.d,
               "uint64": mv.u64, "int64": mv.i64,
               "utilization_level": mv.lvl}.get(kind, mv.d)
        return {"value": val, "kind": kind, "passes": sets.numSets,
                "events": dict(acc), "dur_ns": t_ns}
