#!/usr/bin/env python
"""Test LatentMASMethod._truncate_past on every cache format, because no earlier run ever executed this function.

Only --sequential_info_only and --latent_only reach _truncate_past, and run.py did not define
those flags in the past. The function also used the transformers 4.x legacy Cache shim, which 5.x
removed. This test checks the contract, which is to keep the LAST N positions, on every cache
format that the code supports. It needs no GPU and no model download.

    python yajat/test_truncate_past.py
"""
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from methods.latent_mas import LatentMASMethod

trunc = LatentMASMethod._truncate_past
slice_ = LatentMASMethod._slice_tensor


class _Dummy:
    """Stand in for a bound self, because _truncate_past needs only _slice_tensor."""
    _slice_tensor = staticmethod(slice_)


D = _Dummy()
# B = batch, H = heads, S = cached positions, HD = head dimension.
B, H, S, HD = 2, 4, 12, 8
KEEP = 5
fails = []


def check(name: str, cond: Any) -> None:
    """Print PASS or FAIL for one check, and record the failures."""
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


def make_tensor(offset: int = 0) -> torch.Tensor:
    """Return a [batch, heads, positions, head_dim] tensor in which position p holds the value p + offset."""
    # The test checks values and not only shapes, because a slice from the wrong end has the right shape.
    t = torch.zeros(B, H, S, HD)
    for p in range(S):
        t[:, :, p, :] = p + offset
    return t


print("modern Cache (.layers, transformers >= ~4.54)")
try:
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer_idx in range(3):
        cache.update(make_tensor(), make_tensor(100), layer_idx)
    out = trunc(D, cache, KEEP)
    check("seq length is KEEP", out.get_seq_length() == KEEP)
    kept = out.layers[0].keys[0, 0, :, 0].tolist()
    check("kept the LAST positions", kept == [float(p) for p in range(S - KEEP, S)])
    check("all layers truncated", all(l.keys.shape[-2] == KEEP for l in out.layers))
    check("values truncated too", out.layers[0].values.shape[-2] == KEEP)
except ImportError:
    print("  SKIP  transformers not importable")


print("key_cache/value_cache Cache (transformers 4.3x-4.5x)")


class LegacyStyleCache:
    """Imitate a transformers 4.3x-4.5x cache, which has parallel key and value lists."""

    def __init__(self, n: int) -> None:
        self.key_cache = [make_tensor() for _ in range(n)]
        self.value_cache = [make_tensor(100) for _ in range(n)]


c = LegacyStyleCache(3)
out = trunc(D, c, KEEP)
check("shape", out.key_cache[0].shape[-2] == KEEP)
check("kept the LAST positions",
      out.key_cache[0][0, 0, :, 0].tolist() == [float(p) for p in range(S - KEEP, S)])
check("values truncated too", out.value_cache[2].shape[-2] == KEEP)


print("pre-Cache legacy tuple-of-tuples")
legacy = tuple((make_tensor(), make_tensor(100)) for _ in range(3))
out = trunc(D, legacy, KEEP)
check("shape", out[0][0].shape[-2] == KEEP)
check("kept the LAST positions",
      out[0][0][0, 0, :, 0].tolist() == [float(p) for p in range(S - KEEP, S)])


print("edge cases")
check("None in -> None out", trunc(D, None, KEEP) is None)
check("keep<=0 -> None", trunc(D, legacy, 0) is None)
over = trunc(D, tuple((make_tensor(), make_tensor()) for _ in range(1)), S + 99)
check("keep > length clamps to length", over[0][0].shape[-2] == S)


print()
if fails:
    print(f"FAILED: {len(fails)} -> {fails}")
    sys.exit(1)
print("all _truncate_past checks passed")
