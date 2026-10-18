# -*- coding: utf-8 -*-
"""Scaling study: structured O(d) interval-Krawczyk image vs dense O(d^2).

Measures per-call wall time of the Krawczyk image evaluation K(X) for both
implementations (same inputs) across input dimensions d, plus an end-to-end
SAFE2-style round trip on random two-class ReLU MLPs.

Outputs results/scaling_timing.csv and results/scaling_summary.json with the
log-log slopes (fit of log(time) vs log(d)) for both implementations.
"""
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verifier_lib as vl  # noqa: E402
import verifier_cnn as vc  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE, 'results')
os.makedirs(RESULT_DIR, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

DS = [256, 512, 1024, 2048, 4096, 8192, 16384]
REPEATS = 20


def make_margin_fn(d, hidden=64):
    """Random 2-class ReLU MLP: f0 - f1 = margin, g(x) = 0 scalar equation."""
    net = nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 2))
    net.eval()
    return net


def measure_image(d):
    X_lo = -torch.rand(d) * 0.5
    X_hi = torch.rand(d) * 0.5
    m = (X_lo + X_hi) / 2.0
    g_m = 0.3
    J_lo = -torch.rand(d)
    J_hi = torch.rand(d)
    J_mid = (J_lo + J_hi) / 2.0
    C = vl.moore_penrose(J_mid)
    # dense
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        vl.krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
    t_dense = (time.perf_counter() - t0) / REPEATS
    # structured
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        vc.krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
    t_struct = (time.perf_counter() - t0) / REPEATS
    return t_struct, t_dense


def measure_roundtrip(d):
    """One SAFE2 iteration (bounds + gradient hull + image) on a random net."""
    net = make_margin_fn(d)
    x0 = torch.rand(d)
    eps = 0.01
    X_lo, X_hi = x0 - eps, x0 + eps
    t0 = time.perf_counter()
    logits_lo, logits_hi, ops = vc.forward_bounds(net, X_lo, X_hi, None)
    m_lo, m_hi = vl.margin_interval(logits_lo, logits_hi, 0)
    m = (X_lo + X_hi) / 2.0
    g_m = vc.margin_at(net, m, 0, None)
    J_lo, J_hi = vc.gradient_margin_hull(net, X_lo, X_hi, 0, ops, None)
    J_mid = (J_lo + J_hi) / 2.0
    C = vl.moore_penrose(J_mid)
    K_lo, K_hi = vc.krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
    dt = time.perf_counter() - t0
    return dt, float((K_hi - K_lo).max().item())


def main():
    rows = []
    for d in DS:
        t_s, t_d = measure_image(d)
        print(f'd={d}: struct={t_s*1e3:.3f}ms dense={t_d*1e3:.3f}ms', flush=True)
        rows.append({'d': d, 't_struct_s': t_s, 't_dense_s': t_d})
    with open(os.path.join(RESULT_DIR, 'scaling_timing.csv'), 'w') as f:
        f.write('d,t_struct_s,t_dense_s\n')
        for r in rows:
            f.write(f"{r['d']},{r['t_struct_s']:.6e},{r['t_dense_s']:.6e}\n")
    # log-log slopes over d in [256..16384]
    logd = np.log([r['d'] for r in rows])
    s_struct = np.polyfit(logd, np.log([r['t_struct_s'] for r in rows]), 1)[0]
    s_dense = np.polyfit(logd, np.log([r['t_dense_s'] for r in rows]), 1)[0]
    # end-to-end round trip on real networks
    roundtrip = {}
    for d in [256, 512, 1024, 2048, 4096]:
        dt, width = measure_roundtrip(d)
        roundtrip[str(d)] = {'t_roundtrip_s': dt, 'K_width': width}
        print(f'd={d}: roundtrip={dt*1e3:.2f}ms', flush=True)
    summary = {'image_slope_struct': float(s_struct), 'image_slope_dense': float(s_dense),
               'roundtrip': roundtrip}
    with open(os.path.join(RESULT_DIR, 'scaling_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('SUMMARY:', json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
