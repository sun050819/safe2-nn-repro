# -*- coding: utf-8 -*-
"""Train a small ReLU MLP on MNIST (subsampled) and save checkpoints."""
import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, 'data')
RESULT_DIR = os.path.join(BASE, 'results')
CKPT_DIR = os.path.join(RESULT_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_mnist(n_train=8000, n_test=300):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    train = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    # deterministic subsample
    g = torch.Generator().manual_seed(0)
    idx_tr = torch.randperm(len(train), generator=g)[:n_train]
    idx_te = torch.randperm(len(test), generator=g)[:n_test]
    x_tr = torch.stack([train[i][0].squeeze(0) for i in idx_tr])
    y_tr = torch.tensor([train[i][1] for i in idx_tr])
    x_te = torch.stack([test[i][0].squeeze(0) for i in idx_te])
    y_te = torch.tensor([test[i][1] for i in idx_te])
    return x_tr, y_tr, x_te, y_te


def train_one(seed, epochs=15, batch_size=128, lr=1e-3):
    torch.manual_seed(seed)
    x_tr, y_tr, x_te, y_te = load_mnist()
    model = MLP([784, 64, 64, 10])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = x_tr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_tr[idx], y_tr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            acc = (model(x_te).argmax(1) == y_te).float().mean().item()
        print(f'[seed {seed}] epoch {ep + 1}/{epochs} loss {tot / n:.4f} test_acc {acc:.4f}', flush=True)
    path = os.path.join(CKPT_DIR, f'mlp_seed{seed}.pt')
    torch.save(model.state_dict(), path)
    with open(os.path.join(RESULT_DIR, 'model_stats.csv'), 'a') as f:
        f.write(f'{seed},{acc:.6f}\n')
    print(f'[seed {seed}] saved -> {path}', flush=True)
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = ap.parse_args()
    with open(os.path.join(RESULT_DIR, 'model_stats.csv'), 'w') as f:
        f.write('seed,test_acc\n')
    for s in args.seeds:
        train_one(s)
