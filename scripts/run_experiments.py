# -*- coding: utf-8 -*-
"""Run the full verification experiment suite (sound dual-certificate SAFE2).

Writes per-seed CSV/JSON into results/ and pooled aggregates:
  results_by_eps_seed{s}.csv, runtime_seed{s}.csv, consistency_seed{s}.json,
  hmax_all.csv, protocol_demo_seed{s}.json,
  aggregated_by_eps.csv, cert_distribution_aggregated.csv,
  hmax_aggregated.json, consistency_aggregated.json, protocol_demo.json,
  run_log.txt
"""
import os
import sys
import json
import argparse
import time
import traceback

import torch
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import MLP, load_mnist  # noqa: E402
import verifier_lib as vl  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE, 'results')
CKPT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
os.makedirs(RESULT_DIR, exist_ok=True)

LOG = open(os.path.join(RESULT_DIR, 'run_log.txt'), 'w', encoding='utf-8')


def log(msg):
    print(msg, flush=True)
    LOG.write(msg + '\n')
    LOG.flush()


SEEDS = [0, 1, 2]
EPS_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
N_SAMPLES = 200
N_HMAX = 50
HMAX_HI = 0.5
HMAX_ITERS = 12
ALPHA_MIN = 0.2
ALPHA_FACTOR = 0.8


def load_models():
    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
    models = {}
    for s in SEEDS:
        model = MLP([784, 64, 64, 10])
        model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f'mlp_seed{s}.pt'),
                                         map_location='cpu'))
        model.eval()
        models[s] = model
    return models, x_te, y_te


def select_samples(model, x_te, y_te, n):
    with torch.no_grad():
        preds = model(x_te).argmax(1)
    idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:n]
    return idx


def run_eps_grid(model, idx, x_te, y_te, seed):
    rows = []
    time_rows = []
    cert_counts = {m: {eps: {'robust': 0, 'exist': 0, 'undecided': 0}
                       for eps in EPS_GRID} for m in ('ibp', 'krawczyk', 'safe2')}
    consistency = {'safe2_robust_total': 0, 'safe2_robust_pgd_broken': 0,
                   'safe2_exist_total': 0, 'safe2_exist_verified': 0,
                   'safe2_exist_root_margins': []}
    n_err = 0
    for eps in EPS_GRID:
        for si in idx:
            x0 = x_te[si]
            c = int(y_te[si].item())
            try:
                t0 = time.perf_counter()
                r_ibp = vl.verify_ibp(model, x0, c, eps)
                t_ibp = time.perf_counter() - t0
                t0 = time.perf_counter()
                r_kr = vl.verify_krawczyk_classical(model, x0, c, eps)
                t_kr = time.perf_counter() - t0
                t0 = time.perf_counter()
                r_s2, it_s2, _, root, root_margin = vl.verify_safe2(
                    model, x0, c, eps, ALPHA_MIN, ALPHA_FACTOR)
                t_s2 = time.perf_counter() - t0
            except Exception as e:
                n_err += 1
                log(f'ERROR seed{seed} sample{int(si)} eps{eps}: {e}')
                traceback.print_exc(file=LOG)
                continue
            for method, r in (('ibp', r_ibp), ('krawczyk', r_kr), ('safe2', r_s2)):
                cert_counts[method][eps][r] += 1
            time_rows.append({'eps': eps, 'ibp_s': t_ibp, 'krawczyk_s': t_kr,
                              'safe2_s': t_s2, 'safe2_iters': it_s2})
            if r_s2 == 'robust':
                consistency['safe2_robust_total'] += 1
                if vl.pgd_attack(model, x0, c, eps, steps=10, restarts=2):
                    consistency['safe2_robust_pgd_broken'] += 1
            if r_s2 == 'exist':
                consistency['safe2_exist_total'] += 1
                if root is not None and root_margin <= 1e-6:
                    consistency['safe2_exist_verified'] += 1
                if root is not None:
                    consistency['safe2_exist_root_margins'].append(float(root_margin))
        for method in ('ibp', 'krawczyk', 'safe2'):
            cc = cert_counts[method][eps]
            rows.append({'eps': eps, 'method': method,
                         'robust': cc['robust'], 'exist': cc['exist'],
                         'undecided': cc['undecided'],
                         'certified_rate': cc['robust'] / len(idx)})
    with open(os.path.join(RESULT_DIR, f'results_by_eps_seed{seed}.csv'), 'w') as f:
        f.write('eps,method,robust,exist,undecided,certified_rate\n')
        for r in rows:
            f.write(f"{r['eps']},{r['method']},{r['robust']},{r['exist']},"
                    f"{r['undecided']},{r['certified_rate']:.4f}\n")
    with open(os.path.join(RESULT_DIR, f'runtime_seed{seed}.csv'), 'w') as f:
        f.write('eps,ibp_s,krawczyk_s,safe2_s,safe2_iters\n')
        for r in time_rows:
            f.write(f"{r['eps']},{r['ibp_s']:.4f},{r['krawczyk_s']:.4f},"
                    f"{r['safe2_s']:.4f},{r['safe2_iters']}\n")
    consistency['safe2_exist_root_margins_mean'] = float(
        np.mean(consistency['safe2_exist_root_margins'])) if consistency['safe2_exist_root_margins'] else None
    consistency['safe2_exist_root_margins_max_abs'] = float(
        np.max(np.abs(consistency['safe2_exist_root_margins']))) if consistency['safe2_exist_root_margins'] else None
    with open(os.path.join(RESULT_DIR, f'consistency_seed{seed}.json'), 'w') as f:
        json.dump(consistency, f, indent=2)
    log(f'[seed {seed}] eps grid done, errors={n_err}, '
        f'exist_verified={consistency["safe2_exist_verified"]}/'
        f'{consistency["safe2_exist_total"]}')
    return rows, consistency


def hmax_for_method(model, x0, c, method, hi=HMAX_HI, iters=HMAX_ITERS):
    lo_h, hi_h = 0.0, hi
    best = 0.0
    for _ in range(iters):
        mid = (lo_h + hi_h) / 2.0
        try:
            if method == 'ibp':
                ok = vl.verify_ibp(model, x0, c, mid) == 'robust'
            elif method == 'krawczyk':
                ok = vl.verify_krawczyk_classical(model, x0, c, mid) == 'robust'
            else:
                ok = vl.verify_safe2(model, x0, c, mid, ALPHA_MIN, ALPHA_FACTOR)[0] == 'robust'
        except Exception:
            ok = False
        if ok:
            best = mid
            lo_h = mid
        else:
            hi_h = mid
    return best


def run_hmax(model, idx, x_te, y_te, seed):
    n_err = 0
    hmax_path = os.path.join(RESULT_DIR, 'hmax_all.csv')
    # idempotent per-seed: drop any previous rows of this seed, keep others
    old_rows = []
    if os.path.exists(hmax_path):
        with open(hmax_path) as f:
            for line in f:
                if line.startswith('seed,'):
                    continue
                if not line.startswith(f'{seed},'):
                    old_rows.append(line.rstrip('\n'))
    with open(hmax_path, 'w') as f:
        f.write('seed,sample,h_ibp,h_krawczyk,h_safe2\n')
        for r_ in old_rows:
            f.write(r_ + '\n')
        for si in idx:
            x0 = x_te[si]
            c = int(y_te[si].item())
            try:
                h_ibp = hmax_for_method(model, x0, c, 'ibp')
                h_kr = hmax_for_method(model, x0, c, 'krawczyk')
                h_s2 = hmax_for_method(model, x0, c, 'safe2')
            except Exception as e:
                n_err += 1
                log(f'ERROR hmax seed{seed} sample{int(si)}: {e}')
                continue
            f.write(f'{seed},{int(si)},{h_ibp:.6f},{h_kr:.6f},{h_s2:.6f}\n')
    log(f'[seed {seed}] h_max done, errors={n_err}')


def protocol_demo(model, x_te, y_te, seed):
    """Classical-only vs SAFE2 width trajectories on one sample."""
    idx = select_samples(model, x_te, y_te, N_SAMPLES)
    best = None
    for si in idx:
        x0 = x_te[si]
        c = int(y_te[si].item())
        for eps in (0.05, 0.10, 0.15):
            try:
                _, _, wh_s2, _, _ = vl.verify_safe2(model, x0, c, eps,
                                                    ALPHA_MIN, ALPHA_FACTOR)
                _, _, wh_cl, _, _ = vl.verify_safe2(model, x0, c, eps,
                                                    alpha_min=1.0, alpha_factor=1.0)
            except Exception:
                continue
            score = (len(wh_s2) if wh_s2 else 0) + (len(wh_cl) if wh_cl else 0)
            if best is None or score > best[0]:
                best = (score, si, eps, wh_s2, wh_cl)
    if best is not None:
        _, si, eps, wh_s2, wh_cl = best
        out = {'sample': int(si), 'eps': float(eps),
               'safe2_widths': wh_s2 or [1.0], 'classical_widths': wh_cl or [1.0]}
        with open(os.path.join(RESULT_DIR, f'protocol_demo_seed{seed}.json'), 'w') as f:
            json.dump(out, f, indent=2)
        return out
    return None


def run_seed(model, x_te, y_te, seed):
    idx = select_samples(model, x_te, y_te, N_SAMPLES)
    log(f'[seed {seed}] n_samples={len(idx)}')
    rows, consistency = run_eps_grid(model, idx, x_te, y_te, seed)
    idx_h = idx[:N_HMAX]
    run_hmax(model, idx_h, x_te, y_te, seed)
    demo = protocol_demo(model, x_te, y_te, seed)
    return {'rows': rows, 'consistency': consistency, 'demo': demo}


def aggregate():
    eps_list = EPS_GRID
    methods = ['ibp', 'krawczyk', 'safe2']
    with open(os.path.join(RESULT_DIR, 'aggregated_by_eps.csv'), 'w') as f:
        f.write('eps,method,rate_mean,rate_std\n')
        for eps in eps_list:
            for m in methods:
                vals = []
                for s in SEEDS:
                    with open(os.path.join(RESULT_DIR, f'results_by_eps_seed{s}.csv')) as fh:
                        for line in fh:
                            p = line.strip().split(',')
                            if len(p) == 6 and p[0] == str(eps) and p[1] == m:
                                vals.append(float(p[5]))
                f.write(f'{eps},{m},{np.mean(vals):.4f},{np.std(vals):.4f}\n')
    with open(os.path.join(RESULT_DIR, 'cert_distribution_aggregated.csv'), 'w') as f:
        f.write('eps,method,robust_mean,exist_mean,undecided_mean\n')
        for eps in eps_list:
            for m in methods:
                r_means, e_means, u_means = [], [], []
                for s in SEEDS:
                    with open(os.path.join(RESULT_DIR, f'results_by_eps_seed{s}.csv')) as fh:
                        for line in fh:
                            p = line.strip().split(',')
                            if len(p) == 6 and p[0] == str(eps) and p[1] == m:
                                r_means.append(float(p[2]))
                                e_means.append(float(p[3]))
                                u_means.append(float(p[4]))
                f.write(f"{eps},{m},{np.mean(r_means):.2f},{np.mean(e_means):.2f},"
                        f"{np.mean(u_means):.2f}\n")
    # h_max pooled
    arr = np.loadtxt(os.path.join(RESULT_DIR, 'hmax_all.csv'), delimiter=',', skiprows=1)
    h_ibp = arr[:, 2].tolist()
    h_kr = arr[:, 3].tolist()
    h_s2 = arr[:, 4].tolist()
    w1 = stats.wilcoxon(h_ibp, h_s2)
    w2 = stats.wilcoxon(h_kr, h_s2)
    summary = {'h_ibp_mean': float(np.mean(h_ibp)), 'h_ibp_median': float(np.median(h_ibp)),
               'h_krawczyk_mean': float(np.mean(h_kr)), 'h_krawczyk_median': float(np.median(h_kr)),
               'h_safe2_mean': float(np.mean(h_s2)), 'h_safe2_median': float(np.median(h_s2)),
               'n_samples': int(len(h_ibp)),
               'wilcoxon_ibp_vs_safe2_p': float(w1.pvalue),
               'wilcoxon_krawczyk_vs_safe2_p': float(w2.pvalue)}
    with open(os.path.join(RESULT_DIR, 'hmax_aggregated.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    # consistency pooled
    cons = {'safe2_robust_total': 0, 'safe2_robust_pgd_broken': 0,
            'safe2_exist_total': 0, 'safe2_exist_verified': 0}
    margins = []
    for s in SEEDS:
        with open(os.path.join(RESULT_DIR, f'consistency_seed{s}.json')) as fh:
            d = json.load(fh)
        cons['safe2_robust_total'] += d['safe2_robust_total']
        cons['safe2_robust_pgd_broken'] += d['safe2_robust_pgd_broken']
        cons['safe2_exist_total'] += d['safe2_exist_total']
        cons['safe2_exist_verified'] += d['safe2_exist_verified']
        margins.extend(d['safe2_exist_root_margins'])
    cons['safe2_exist_root_margins_mean'] = float(np.mean(margins)) if margins else None
    cons['safe2_exist_root_margins_max_abs'] = float(np.max(np.abs(margins))) if margins else None
    with open(os.path.join(RESULT_DIR, 'consistency_aggregated.json'), 'w') as f:
        json.dump(cons, f, indent=2)
    # protocol demo: pick the seed demo with most iterations
    best = None
    for s in SEEDS:
        p = os.path.join(RESULT_DIR, f'protocol_demo_seed{s}.json')
        if os.path.exists(p):
            with open(p) as fh:
                d = json.load(fh)
            if best is None or len(d['safe2_widths']) > len(best['safe2_widths']):
                best = d
    if best:
        with open(os.path.join(RESULT_DIR, 'protocol_demo.json'), 'w') as f:
            json.dump(best, f, indent=2)
    return summary, cons


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=-1, help='single seed (default: all)')
    ap.add_argument('--stage', choices=['eps', 'hmax', 'demo', 'aggregate', 'all'],
                    default='all')
    args = ap.parse_args()
    models, x_te, y_te = load_models()
    seeds = SEEDS if args.seed < 0 else [args.seed]
    if args.stage == 'aggregate':
        summary, cons = aggregate()
        log('AGGREGATED:')
        log(json.dumps({'hmax': summary, 'consistency': cons}, indent=2, default=str))
        log('ALL DONE')
        LOG.close()
        sys.exit(0)
    if args.stage in ('all', 'eps', 'hmax'):
        if not os.path.exists(os.path.join(RESULT_DIR, 'hmax_all.csv')):
            with open(os.path.join(RESULT_DIR, 'hmax_all.csv'), 'w') as f:
                f.write('seed,sample,h_ibp,h_krawczyk,h_safe2\n')
    for s in seeds:
        model = models[s]
        idx = select_samples(model, x_te, y_te, N_SAMPLES)
        log(f'[seed {s}] n_samples={len(idx)}')
        if args.stage in ('all', 'eps'):
            run_eps_grid(model, idx, x_te, y_te, s)
        if args.stage in ('all', 'hmax'):
            run_hmax(model, idx[:N_HMAX], x_te, y_te, s)
        if args.stage in ('all', 'demo'):
            protocol_demo(model, x_te, y_te, s)
    if args.stage == 'all':
        summary, cons = aggregate()
        log('AGGREGATED:')
        log(json.dumps({'hmax': summary, 'consistency': cons}, indent=2, default=str))
        log('ALL DONE')
    LOG.close()
