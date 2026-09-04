#!/usr/bin/env python
"""Unit test for LatentMASMethod._truncate_past across cache representations.

_truncate_past is the one function in the repo that had never executed: it is
reachable only from --sequential_info_only / --latent_only, which run.py did not
define until now. It also used the transformers 4.x legacy Cache shim that 5.x
removed. This pins the contract -- keep the LAST N positions -- on every cache
shape the code claims to support, with no GPU and no model download.

    python yajat/test_truncate_past.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from methods.latent_mas import LatentMASMethod

trunc = LatentMASMethod._truncate_past
slice_ = LatentMASMethod._slice_tensor


class _Dummy:
    """Stand-in for a bound self; _truncate_past only needs _slice_tensor."""
    _slice_tensor = staticmethod(slice_)


D = _Dummy()
B, H, S, HD = 2, 4, 12, 8
KEEP = 5
fails = []


def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    if not cond:
        fails.append(name)


def make_tensor(offset=0):
    # position p is filled with the value p so truncation is verifiable by value,
    # not just by shape -- a slice from the wrong end still has the right shape.
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
    def __init__(self, n):
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
