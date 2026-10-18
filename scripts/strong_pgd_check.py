# -*- coding: utf-8 -*-
"""Stronger PGD falsification of every SAFE2 robust certificate.

The manuscript's grid runs used a short falsification attack (10 steps,
2 restarts).  This script re-checks every robust certificate issued by
SAFE2-Krawczyk with a substantially stronger attack geometry:

  MNIST MLPs:   100 steps, 20 restarts
  CIFAR-10 CNNs: 50 steps, 10 restarts

A certificate is contradicted only if the stronger attack finds an
adversarial example inside the certified box.  Output:
results/strong_pgd_check.json
"""
import os
import sys
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import MLP, load_mnist  # noqa: E402
from train_cnn import make_cnn  # noqa: E402
import verifier_lib as vl  # noqa: E402
import verifier_cnn as vc  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE, 'results')
CKPT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
DATA_DIR = os.path.join(BASE, 'data')
INPUT_SHAPE = (3, 32, 32)

SEEDS = [0, 1, 2]
N_SAMPLES = 200
MNIST_EPS = 0.01


def mnist_robust_samples():
    x_te, y_te = load_mnist(n_train=8000, n_test=300)[2:]
    robust = []
    for s in SEEDS:
        model = MLP([784, 64, 64, 10])
        model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f'mlp_seed{s}.pt'),
                                         map_location='cpu'))
        model.eval()
        with torch.no_grad():
            preds = model(x_te).argmax(1)
        idx = (preds == y_te).nonzero(as_tuple=False).flatten()[:N_SAMPLES]
        for si in idx:
            x0 = x_te[si]
            c = int(y_te[si].item())
            r, _, _, _, _ = vl.verify_safe2(model, x0, c, MNIST_EPS, 0.2, 0.8)
            if r == 'robust':
                robust.append((model, x0, c, s))
    return robust


def check_mnist():
    rob = mnist_robust_samples()
    broken = 0
    for k, (model, x0, c, s) in enumerate(rob):
        if vl.pgd_attack(model, x0, c, MNIST_EPS, steps=100, restarts=20,
                         step_size=0.01):
            broken += 1
        if (k + 1) % 10 == 0:
            print(f'[mnist strong pgd] {k + 1}/{len(rob)} broken={broken}',
                  flush=True)
    return {'total': len(rob), 'broken': broken}


def cnn_robust_samples():
    import torchvision
    import torchvision.transforms as transforms
    tf = transforms.Compose([transforms.ToTensor()])
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=False,
                                        transform=tf)
    x_te = torch.stack([test[i][0] for i in range(len(test))])
    y_te = torch.tensor([test[i][1] for i in range(len(test))])
    robust = []
    for reg in ('std', 'ibp'):
        for s in SEEDS:
            model = make_cnn()
            model.load_state_dict(torch.load(
                os.path.join(CKPT_DIR, f'cnn_{reg}_seed{s}.pt'),
                map_location='cpu'))
            model.eval()
            verdicts = {}
            p = os.path.join(RESULT_DIR, f'cnn_verdicts_{reg}_seed{s}.csv')
            if not os.path.exists(p):
                continue
            with open(p) as f:
                for line in f:
                    q = line.strip().split(',')
                    if len(q) != 4 or q[0] == 'eps':
                        continue
                    verdicts[(float(q[0]), q[1], q[2])] = q[3]
            for (eps, sample, method), v in verdicts.items():
                if method == 'safe2' and v == 'robust':
                    si = int(sample)
                    x0 = x_te[si].flatten()
                    c = int(y_te[si].item())
                    robust.append((model, x0, c, eps, s))
    return robust


def check_cnn():
    rob = cnn_robust_samples()
    broken = 0
    for k, (model, x0, c, eps, s) in enumerate(rob):
        if vc.pgd_attack(model, x0, c, eps, INPUT_SHAPE, steps=50, restarts=10,
                         step_size=0.02):
            broken += 1
        if (k + 1) % 200 == 0:
            print(f'[cnn strong pgd] {k + 1}/{len(rob)} broken={broken}',
                  flush=True)
    return {'total': len(rob), 'broken': broken}


if __name__ == '__main__':
    m = check_mnist()
    c = check_cnn()
    out = {'mnist': m, 'cnn': c}
    with open(os.path.join(RESULT_DIR, 'strong_pgd_check.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('STRONG PGD:', json.dumps(out, indent=2))
    print('ALL DONE')
