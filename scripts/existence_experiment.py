# -*- coding: utf-8 -*-
"""Existence-certificate validation on a low-dimensional ReLU classifier.

On MNIST-scale inputs, 2-pixel slices almost never contain a sign change of
the margin (reported as a negative finding in the manuscript). To validate the
existence-certificate mechanism of Theorem 2 we use a 2-D ReLU MLP (two moons)
whose decision boundary provably crosses the verification boxes: for boxes
with a sign change at the corners, we compare the classical iteration and the
SAFE2 gated relaxation in terms of strict-inclusion (root-existence)
certificates and extracted-root margins.

Usage:
  python existence_experiment.py --train     # train & save the model
  python existence_experiment.py --verify    # run the verification study
Outputs: results/existence_model.pt, results/existence_results.csv,
         results/existence_aggregated.json
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verifier_lib as vl  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE, 'results')
CKPT = os.path.join(RESULT_DIR, 'existence_model.pt')

SEED = 0
N_TRAIN = 2000
N_TEST = 500
EPS_GRID = [0.30, 0.50]
N_SAMPLES = 20
ALPHA_MIN = 0.5
ALPHA_FACTOR = 0.5
MAX_IT_R = 15
MAX_IT_E = 12


class TwoMoonNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def make_moons(n_samples, noise=0.10, random_state=0):
    """Deterministic two-moons generator (sklearn geometry)."""
    rng = np.random.RandomState(random_state)
    n = n_samples // 2
    t = np.linspace(0, np.pi, n)
    outer_x, outer_y = np.cos(t), np.sin(t)
    inner_x, inner_y = 1 - np.cos(t), 1 - np.sin(t) - 0.5
    X = np.vstack([np.hstack([outer_x, inner_x]),
                   np.hstack([outer_y, inner_y])]).T
    y = np.hstack([np.zeros(n), np.ones(n)])
    X = X + rng.normal(scale=noise, size=X.shape)
    return X.astype(np.float32), y.astype(np.int64)


def make_data(seed):
    X, y = make_moons(N_TRAIN + N_TEST, noise=0.10, random_state=seed)
    perm = np.random.RandomState(seed + 1).permutation(len(X))
    X, y = X[perm], y[perm]
    Xtr, ytr = torch.from_numpy(X[:N_TRAIN]), torch.from_numpy(y[:N_TRAIN])
    Xte, yte = torch.from_numpy(X[N_TRAIN:]), torch.from_numpy(y[N_TRAIN:])
    return Xtr, ytr, Xte, yte


def train_model(Xtr, ytr):
    torch.manual_seed(SEED)
    model = TwoMoonNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for epoch in range(300):
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            out = model(Xtr[b])
            loss = lossf(out, ytr[b])
            loss.backward()
            opt.step()
    model.eval()
    torch.save(model.state_dict(), CKPT)
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    print(f'train done, test acc = {acc:.4f}')


def margin_at(model, x, c):
    with torch.no_grad():
        logits = model(x.unsqueeze(0)).squeeze(0)
    others = [j for j in range(2) if j != c]
    return float((logits[c] - torch.max(logits[others])).item())


def corner_min(model, c, eps):
    vals = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            x = torch.tensor([sx * eps, sy * eps])
            vals.append(margin_at(model, x, c))
    return min(vals)


def verify():
    Xtr, ytr, Xte, yte = make_data(SEED)
    model = TwoMoonNet()
    model.load_state_dict(torch.load(CKPT, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1)
    # samples near the boundary: correct-classified with small margin
    margins = []
    for i in range(len(Xte)):
        if preds[i] != yte[i]:
            continue
        m = margin_at(model, Xte[i], int(yte[i]))
        margins.append((abs(m), i))
    margins.sort()
    idx = [i for _, i in margins[:N_SAMPLES]]
    csv_path = os.path.join(RESULT_DIR, 'existence_results.csv')
    with open(csv_path, 'w') as f:
        f.write('sample,eps,mid_margin,corner_min,nontrivial,classical_iter,safe2,root_margin_cl,root_margin\n')
    total = N_SAMPLES * len(EPS_GRID)
    done = 0
    for si in idx:
        x0 = Xte[si]
        c = int(yte[si])
        for eps in EPS_GRID:
            g_m = margin_at(model, x0, c)
            cmin = corner_min(model, c, eps)
            nontrivial = 1 if (g_m > 0 and cmin <= 0) else 0
            r_cl, _, _, root_cl, rm_cl = vl.verify_safe2(model, x0, c, eps,
                                                          alpha_min=1.0, alpha_factor=1.0,
                                                          max_iter_r=MAX_IT_R, max_iter_e=MAX_IT_E)
            r_s2, _, _, root, rm = vl.verify_safe2(model, x0, c, eps,
                                                   ALPHA_MIN, ALPHA_FACTOR,
                                                   max_iter_r=MAX_IT_R, max_iter_e=MAX_IT_E)
            rmc = '' if root_cl is None else f"{float(rm_cl):.6f}"
            rmf = '' if root is None else f"{float(rm):.6f}"
            with open(csv_path, 'a') as f:
                f.write(f'{int(si)},{eps},{g_m:.4f},{cmin:.4f},{nontrivial},'
                        f'{r_cl},{r_s2},{rmc},{rmf}\n')
            done += 1
            if done % 20 == 0:
                print(f'[existence] {done}/{total}', flush=True)
    # aggregate (restricted to nontrivial sign-change boxes)
    rows = []
    with open(csv_path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) == 9 and p[0] != 'sample':
                rows.append(p)
    out = {}
    for eps in EPS_GRID:
        sel = [p for p in rows if float(p[1]) == eps]
        nt = [p for p in sel if p[4] == '1']
        n_ex_cl = sum(1 for p in nt if p[5] == 'exist')
        n_ex_s2 = sum(1 for p in nt if p[6] == 'exist')
        m_cl = [float(p[7]) for p in nt if p[5] == 'exist' and p[7] != '']
        m_s2 = [float(p[8]) for p in nt if p[6] == 'exist' and p[8] != '']
        out[eps] = {
            'n_samples': len(sel),
            'nontrivial_boxes': len(nt),
            'exist_classical': n_ex_cl,
            'exist_safe2': n_ex_s2,
            'classical_success_on_nontrivial': (n_ex_cl / len(nt)) if nt else None,
            'safe2_success_on_nontrivial': (n_ex_s2 / len(nt)) if nt else None,
            'root_margin_cl_mean': float(np.mean(m_cl)) if m_cl else None,
            'root_margin_cl_max_abs': float(np.max(np.abs(m_cl))) if m_cl else None,
            'root_margin_safe2_mean': float(np.mean(m_s2)) if m_s2 else None,
            'root_margin_safe2_max_abs': float(np.max(np.abs(m_s2))) if m_s2 else None,
        }
    with open(os.path.join(RESULT_DIR, 'existence_aggregated.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('EXISTENCE DONE')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--verify', action='store_true')
    args = ap.parse_args()
    Xtr, ytr, Xte, yte = make_data(SEED)
    if args.train:
        train_model(Xtr, ytr)
    if args.verify:
        verify()
