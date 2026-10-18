# -*- coding: utf-8 -*-
"""SAFE2-assisted BaB: compare pure BaB vs SAFE2 preprocessing + BaB.

On the two-moons benchmark, compare:
1. Pure BaB: run BaB on every sample, count total nodes.
2. SAFE2-assisted: run SAFE2 first; if SAFE2 gives a verdict, skip BaB;
   if SAFE2 is undecided, run BaB. Count total BaB nodes.

Writes results/safe2_assisted_bab.json.
"""
import os
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verifier_lib as vl
import verifier_bab as vb
from existence_experiment import TwoMoonNet

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE, 'results')
CKPT = os.path.join(RESULT_DIR, 'existence_model.pt')

SEED = 0
N_SAMPLES = 20
EPS = 0.3
BAB_BUDGET = 500

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model
    model = TwoMoonNet()
    model.load_state_dict(torch.load(CKPT, map_location='cpu'))
    model.eval()

    # Generate test samples
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=500, noise=0.1, random_state=SEED)
    X = torch.tensor(X, dtype=torch.float32)

    # Pick nontrivial samples (midpoint robust but corner violates)
    nontrivial = []
    for i in range(len(X)):
        x0 = X[i]
        logits = model(x0).squeeze()
        margin = logits[0] - logits[1]  # margin for class 0 vs class 1
        # Check corners
        corners = [x0 + EPS * torch.tensor([sx, sy], dtype=torch.float32)
                   for sx in [-1, 1] for sy in [-1, 1]]
        corner_margins = []
        for c in corners:
            l = model(c).squeeze()
            corner_margins.append(l[0] - l[1])
        # Nontrivial: midpoint is robust, but some corner violates
        if margin.item() > 0 and min(corner_margins).item() < 0:
            nontrivial.append((i, x0))
        if len(nontrivial) >= N_SAMPLES:
            break

    print(f'Found {len(nontrivial)} nontrivial samples')

    # Experiment 1: Pure BaB
    print('Running pure BaB...')
    pure_bab_nodes = []
    pure_bab_verdicts = []
    for idx, (i, x0) in enumerate(nontrivial):
        t0 = time.time()
        verdict, nodes, _ = vb.verify_split_bab(model, x0, 0, EPS,
                                                 node_budget=BAB_BUDGET)
        dt = time.time() - t0
        pure_bab_nodes.append(nodes)
        pure_bab_verdicts.append(verdict)
        if (idx + 1) % 5 == 0:
            print(f'  {idx+1}/{len(nontrivial)}: nodes={nodes}, verdict={verdict}')

    # Experiment 2: SAFE2-assisted BaB
    print('Running SAFE2-assisted BaB...')
    safe2_bab_nodes = []
    safe2_verdicts = []
    safe2_skip = 0
    for idx, (i, x0) in enumerate(nontrivial):
        # First run SAFE2
        cert, iters, _, root, root_margin = vl.verify_safe2(
            model, x0, 0, EPS, alpha_min=0.5, alpha_factor=0.5,
            max_iter_r=12, max_iter_e=12)

        if cert in ('robust', 'exist', 'not_robust'):
            # SAFE2 gave a verdict, skip BaB
            safe2_bab_nodes.append(0)
            safe2_verdicts.append(cert)
            safe2_skip += 1
        else:
            # SAFE2 undecided, run BaB
            t0 = time.time()
            verdict, nodes, _ = vb.verify_split_bab(model, x0, 0, EPS,
                                                     node_budget=BAB_BUDGET)
            dt = time.time() - t0
            safe2_bab_nodes.append(nodes)
            safe2_verdicts.append(verdict)

        if (idx + 1) % 5 == 0:
            print(f'  {idx+1}/{len(nontrivial)}: safe2={cert}, bab_nodes={safe2_bab_nodes[-1]}')

    # Summary
    total_pure = sum(pure_bab_nodes)
    total_safe2 = sum(safe2_bab_nodes)
    saved = total_pure - total_safe2
    pct = 100 * saved / max(total_pure, 1)

    result = {
        'n_samples': len(nontrivial),
        'eps': EPS,
        'bab_budget': BAB_BUDGET,
        'pure_bab_total_nodes': total_pure,
        'safe2_assisted_total_nodes': total_safe2,
        'nodes_saved': saved,
        'pct_saved': pct,
        'safe2_skip_count': safe2_skip,
        'pure_bab_verdicts': pure_bab_verdicts,
        'safe2_verdicts': safe2_verdicts,
        'pure_bab_nodes_per_sample': pure_bab_nodes,
        'safe2_bab_nodes_per_sample': safe2_bab_nodes,
    }

    out_path = os.path.join(RESULT_DIR, 'safe2_assisted_bab.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n=== Results ===')
    print(f'Pure BaB total nodes: {total_pure}')
    print(f'SAFE2-assisted total nodes: {total_safe2}')
    print(f'Nodes saved: {saved} ({pct:.1f}%)')
    print(f'SAFE2 skipped BaB on {safe2_skip}/{len(nontrivial)} samples')
    print(f'Results saved to {out_path}')

if __name__ == '__main__':
    main()
