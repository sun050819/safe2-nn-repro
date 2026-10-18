# -*- coding: utf-8 -*-
"""Exact cross-validation of SAFE2 certificates against a complete BaB solver.

Two workloads:
  (a) random small ReLU MLPs (d in {4,6}, h=8): SAFE2 (verifier_lib, dual
      certificates) vs the complete neuron-branching BaB (verify_bab), on a
      per-sample epsilon grid; every SAFE2 verdict that BaB can decide must
      agree, and every BaB counterexample must be certified (true margin <= 0).
  (b) the two-moons benchmark: SAFE2 vs the complete input-splitting BaB
      (verify_split_bab) with a modest node budget; agreement is reported on
      the cases the complete verifier can decide within budget.

Writes results/bab_crossval.json (programmatic, used by the paper generator).
"""
import os
import sys
import json
import random

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verifier_bab as vb            # noqa: E402
import verifier_cnn as vc            # noqa: E402
import verifier_lib as vl            # noqa: E402
from train import MLP                # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE, 'results')
CKPT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
os.makedirs(RESULT_DIR, exist_ok=True)

SEEDS = [0, 1, 2]
# (d, h, k, eps, ntrials) per radius: small radii stress SAFE2 robust
# certificates (confirmed by BaB in few nodes); large radii mostly leave
# SAFE2 undecided, quantifying the coverage gap.
RANDOM_NETS = [(4, 8, 3, 0.05, 12), (4, 8, 3, 0.10, 12), (4, 8, 3, 0.30, 20),
               (6, 8, 3, 0.05, 8), (6, 8, 3, 0.10, 8), (6, 8, 3, 0.20, 12)]
MLP_EPS = [0.01, 0.02, 0.05]
BAB_BUDGET_RANDOM = 50000
BAB_BUDGET_MOONS = 500


def make_net(d, h, k):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(),
                                     nn.Linear(h, h), nn.ReLU(),
                                     nn.Linear(h, k))

        def forward(self, x):
            return self.net(x)
    return Net()


def map_verdict_s2(r):
    # SAFE2 returns (certificate, iters, widths, root, root_margin)
    cert = r[0] if isinstance(r, (tuple, list)) else r
    if cert == 'robust':
        return 'robust'
    if cert == 'exist':
        return 'not_robust'
    return 'undecided'


def check_random_nets():
    rng = np.random.RandomState(0)
    torch.manual_seed(0)
    rows = []
    for (d, h, k, eps, ntrials) in RANDOM_NETS:
        for t in range(ntrials):
            net = make_net(d, h, k)
            net.eval()
            x0 = torch.tensor(rng.rand(d), dtype=torch.float32) * 0.5 + 0.2
            c = 0
            r_s2 = vl.verify_safe2(net, x0, c, eps)
            v_s2 = map_verdict_s2(r_s2)
            v_bab, nodes, ce = vb.verify_bab(net, x0, c, eps, None,
                                             node_budget=BAB_BUDGET_RANDOM)
            agreed = None
            if v_bab in ('robust', 'not_robust'):
                agreed = int(v_s2 == v_bab)
            rows.append({'net': f'd{d}h{h}', 'trial': t, 'eps': eps,
                         'safe2': v_s2, 'bab': v_bab, 'nodes': nodes,
                         'agreed': agreed})
            print(f'[random d{d}h{h} t{t}] eps={eps} safe2={v_s2} bab={v_bab} '
                  f'nodes={nodes}', flush=True)
    return rows


def check_two_moons():
    from existence_experiment import TwoMoonNet, make_moons
    model = TwoMoonNet()
    p = os.path.join(CKPT_DIR, 'existence_model.pt')
    if not os.path.exists(p):
        p = os.path.join(RESULT_DIR, 'existence_model.pt')
    model.load_state_dict(torch.load(p, map_location='cpu'))
    model.eval()
    X, y = make_moons(500, noise=0.10, random_state=0)
    X = torch.tensor(X)
    y = torch.tensor(y)
    with torch.no_grad():
        preds = model(X).argmax(1)
    idx = (preds == y).nonzero(as_tuple=False).flatten()[:8]
    rows = []
    for si in idx:
        x0 = X[si]
        c = int(y[si].item())
        for eps in (0.1, 0.2, 0.3, 0.5):
            r_s2 = vl.verify_safe2(model, x0, c, eps)
            v_s2 = map_verdict_s2(r_s2)
            v_bab, nodes, ce = vb.verify_split_bab(model, x0, c, eps,
                                                   node_budget=BAB_BUDGET_MOONS)
            agreed = None
            if v_bab in ('robust', 'not_robust'):
                agreed = int(v_s2 == v_bab)
            rows.append({'net': 'twomoons', 'sample': int(si), 'eps': eps,
                         'safe2': v_s2, 'bab': v_bab, 'nodes': nodes,
                         'agreed': agreed})
            print(f'[moons s{int(si)} eps={eps}] safe2={v_s2} bab={v_bab} '
                  f'nodes={nodes}', flush=True)
    return rows


def main():
    random_rows = check_random_nets()
    moons_rows = check_two_moons()
    all_rows = random_rows + moons_rows

    def tally(rows):
        d = {'n_total': len(rows), 'n_decided': 0, 'n_agree': 0,
             'n_disagree': 0, 'n_gap': 0}
        for r in rows:
            if r['bab'] not in ('robust', 'not_robust'):
                continue
            d['n_decided'] += 1
            if r['safe2'] == 'undecided':
                d['n_gap'] += 1            # SAFE2 silent; not a contradiction
            elif r['safe2'] == r['bab']:
                d['n_agree'] += 1
            else:
                d['n_disagree'] += 1       # a false certificate: must be 0
        return d

    out = {'random_nets': tally(random_rows),
           'two_moons': tally(moons_rows),
           'overall': tally(all_rows),
           'rows': all_rows}
    with open(os.path.join(RESULT_DIR, 'bab_crossval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('CROSSVAL:', json.dumps(out, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
