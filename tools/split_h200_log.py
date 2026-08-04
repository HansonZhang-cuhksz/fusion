#!/usr/bin/env python3
"""Split log/run_h200/f04f05.log into per-fusion logs and quarantine the compiler spam.

WHY THIS EXISTS. The f04f05 log is 107.7 MB, over GitHub's 100 MiB limit, and it is
currently .gitignored -- so the record of the study's most-measured family is not in the
repository at all. Splitting it by fusion alone would give two ~54 MB files, because the
size is not F4-vs-F5 volume: **93 % of the lines are Triton compiler output**, dominated by
one internal assertion

    WSLowerToken.cpp:73 processProducerCommitOp(...): Assertion `false' failed
    <kernel>.py:NNN:0: Pipeline failed while executing [NVGPUWarpSpecialization]

repeated once per attempted config, each followed by an MLIR reproducer whose `pipeline:`
string alone is ~2.5 KB, plus a full TTGIR dump.

That output is a RESULT, not noise -- it is the record of Triton's warp-specialization pass
failing to compile these kernels on sm_90 -- so it is preserved, compressed, rather than
discarded. What it must not do is keep the readable log out of git.

Outputs (default, next to the input):
    f04.log                     readable log, #4 variants + the shared baselines
    f05.log                     readable log, #5 variants + the shared baselines
    f04f05_compiler.log.gz      every compiler assertion / reproducer / IR line, gzipped

Usage:
    python3 tools/split_h200_log.py [log/run_h200/f04f05.log] [--outdir DIR] [--keep-uncompressed]
"""
from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

# The readable log is an ALLOWLIST, not a blocklist.
#
# Blocklisting the compiler output was tried first and leaked: MLIR attribute definitions
# (`#blocked = #ttg.blocked<...>`), module terminators (`#-}`) and the Python source echoes
# inside a traceback all slipped through, and each new leak needs another pattern. The
# readable log, by contrast, has a tiny fixed grammar -- 189 lines out of 1,240,800 in the
# f04f05 log, i.e. 0.015 % -- so recognising IT and quarantining everything else is both
# shorter and safe by default: an unrecognised line goes to the compiler stream, where it is
# preserved, rather than into a "readable" log it might corrupt.
READABLE = (
    re.compile(r"^#\s"),                       # `# /usr/bin/python ...`, `# started ...`
    re.compile(r"^\s*\["),                      # [harness] [hw] [env] [screen X] [F5] [U5] ...
    re.compile(r"^\s*==\s"),                    # regime banners
    re.compile(r"^\s*==\s*\w+"),
    re.compile(r"^\s+grids:"),
    re.compile(r"^\s*regime\s"),                # summary table header
    re.compile(r"^wrote\s"),
    re.compile(r"^\s*(decode_bs|prefill_t)\w*\s+[-\d]"),   # summary table rows
)

# Variant routing. BOTH arms of each pair carry the family letter: `[F4]`/`[F4+topk]` are the
# fused arms and `[U4]`/`[U4+topk]` the unfused ones, so a per-fusion log has to take both or
# it holds a speedup with no denominator. Untagged lines (header, regime banners, and the
# shared baselines `[norm]` `[add+norm]` `[router]` `[topk]` `[grid ...]`) go to BOTH files,
# because each fusion is scored against those and a log without them cannot be audited alone.
F4 = re.compile(r"\[\s*(screen\s+)?[FU]4(\+topk)?\s*\]")
F5 = re.compile(r"\[\s*(screen\s+)?[FU]5(\+topk)?\s*\]")


def is_readable(line: str) -> bool:
    return any(p.search(line) for p in READABLE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default="log/run_h200/f04f05.log")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--keep-uncompressed", action="store_true")
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.outdir) if a.outdir else src.parent
    out.mkdir(parents=True, exist_ok=True)

    p4, p5 = out / "f04.log", out / "f05.log"
    pc = out / "f04f05_compiler.log"
    n = {"total": 0, "compiler": 0, "f4": 0, "f5": 0, "both": 0}

    hdr = (f"# split from {src.name} by tools/split_h200_log.py -- readable lines only.\n"
           f"# Triton compiler assertions, MLIR reproducers and IR dumps live in "
           f"{pc.name}.gz\n")

    with src.open(errors="replace") as fh, \
            p4.open("w") as f4, p5.open("w") as f5, pc.open("w") as fc:
        f4.write(hdr.replace("f04f05.log by", "f04f05.log (fusion #4 view) by"))
        f5.write(hdr.replace("f04f05.log by", "f04f05.log (fusion #5 view) by"))
        for line in fh:
            n["total"] += 1
            if not is_readable(line):
                n["compiler"] += 1
                fc.write(line)
                continue
            in4, in5 = bool(F4.search(line)), bool(F5.search(line))
            if in4 and not in5:
                n["f4"] += 1
                f4.write(line)
            elif in5 and not in4:
                n["f5"] += 1
                f5.write(line)
            else:                       # shared context: header, regime banner, baselines
                n["both"] += 1
                f4.write(line)
                f5.write(line)

    raw = pc.read_bytes()
    with gzip.open(f"{pc}.gz", "wb", compresslevel=9) as gz:
        gz.write(raw)
    if not a.keep_uncompressed:
        pc.unlink()

    mb = lambda p: Path(p).stat().st_size / 2**20  # noqa: E731
    print(f"  source            {src}  {mb(src):8.1f} MB  {n['total']:>9,} lines")
    print(f"  -> {p4.name:<22}{mb(p4):8.1f} MB  {n['f4'] + n['both']:>9,} lines")
    print(f"  -> {p5.name:<22}{mb(p5):8.1f} MB  {n['f5'] + n['both']:>9,} lines")
    print(f"  -> {pc.name + '.gz':<22}{mb(str(pc) + '.gz'):8.1f} MB  "
          f"{n['compiler']:>9,} lines  ({100 * n['compiler'] / max(n['total'], 1):.0f}% of source)")
    print(f"     shared lines copied to both readable logs: {n['both']:,}")


if __name__ == "__main__":
    main()
