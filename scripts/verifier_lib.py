# -*- coding: utf-8 -*-
"""
SAFE2-Krawczyk verification library.

Implements:
  - interval arithmetic on torch tensors (sound outward-rounding semantics)
  - interval bound propagation (IBP) forward bounds for ReLU MLPs
  - interval gradient propagation (backward) for the margin function
  - the classical interval-Krawczyk operator with Moore-Penrose preconditioner
  - the SAFE2 gated relaxed protocol with dual certificates
     (robust certificate / root-existence certificate / undecided)

Everything here is deterministic and reproducible given fixed seeds.
"""
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Interval arithmetic (intervals represented as (lo, hi) tensors)
# ---------------------------------------------------------------------------

def _outward(lo, hi):
    """Outward rounding of interval endpoints.

    Every floating-point interval operation must return an enclosure of the
    exact real interval: each endpoint is extended by one ulp toward the
    corresponding infinity so that round-to-nearest errors of the endpoint
    arithmetic cannot make the enclosure unsound.
    """
    return (torch.nextafter(lo, torch.full_like(lo, float('-inf'))),
            torch.nextafter(hi, torch.full_like(hi, float('inf'))))


def iadd(l1, h1, l2, h2):
    return _outward(l1 + l2, h1 + h2)


def isub(l1, h1, l2, h2):
    return _outward(l1 - h2, h1 - l2)


def imul(l1, h1, l2, h2):
    a = l1 * l2
    b = l1 * h2
    c = h1 * l2
    d = h1 * h2
    lo = torch.minimum(torch.minimum(a, b), torch.minimum(c, d))
    hi = torch.maximum(torch.maximum(a, b), torch.maximum(c, d))
    return _outward(lo, hi)


def irelu(l, h):
    return torch.clamp(l, min=0.0), torch.clamp(h, min=0.0)


def linear_bounds(W, b, x_lo, x_hi):
    """Sound interval bounds of y = W x + b over the box [x_lo, x_hi]."""
    Wp = torch.clamp(W, min=0.0)
    Wn = torch.clamp(W, max=0.0)
    y_lo = Wp @ x_lo + Wn @ x_hi + b
    y_hi = Wp @ x_hi + Wn @ x_lo + b
    return _outward(y_lo, y_hi)


def interval_matvec(M_lo, M_hi, x_lo, x_hi):
    """Interval matrix-vector product via corner evaluation (sound over-approx)."""
    p1 = M_lo @ x_lo
    p2 = M_lo @ x_hi
    p3 = M_hi @ x_lo
    p4 = M_hi @ x_hi
    lo = torch.minimum(torch.minimum(p1, p2), torch.minimum(p3, p4))
    hi = torch.maximum(torch.maximum(p1, p2), torch.maximum(p3, p4))
    return _outward(lo, hi)


def empty_intersection(X_lo, X_hi, K_lo, K_hi):
    """True iff the interval boxes X and K have empty intersection."""
    return torch.any((K_hi < X_lo) | (K_lo > X_hi))


def strict_inclusion(X_lo, X_hi, K_lo, K_hi):
    """True iff K is strictly contained in X (componentwise, strict)."""
    return bool(torch.all(K_lo > X_lo).item()) and bool(torch.all(K_hi < X_hi).item())


def strict_shrink(X_lo, X_hi, K_lo, K_hi):
    """True iff K is nonempty and its mean width is strictly smaller than X's.
    (Coordinate-wise shrink is too strict in high dimension.)"""
    if empty_intersection(X_lo, X_hi, K_lo, K_hi):
        return False
    w_new = (K_hi - K_lo).abs().mean().item()
    w_old = (X_hi - X_lo).abs().mean().item()
    return w_new < w_old


def box_width(X_lo, X_hi):
    return float((X_hi - X_lo).abs().max().item())


# ---------------------------------------------------------------------------
# Forward interval bounds and gradient intervals for a ReLU MLP
# ---------------------------------------------------------------------------

def forward_bounds(model, x_lo, x_hi):
    """Returns (logit_lo, logit_hi, ops) where ops records (kind, preact_bounds)
    for each module in order; preact bounds are attached to linear layers."""
    ops = []
    xl, xh = x_lo, x_hi
    for mod in model.net:
        if isinstance(mod, torch.nn.Linear):
            xl, xh = linear_bounds(mod.weight, mod.bias, xl, xh)
            ops.append(('linear', (xl, xh)))
        elif isinstance(mod, torch.nn.ReLU):
            xl, xh = irelu(xl, xh)
            ops.append(('relu', None))
    return xl, xh, ops


def gradient_interval_logit(model, x_lo, x_hi, k, ops):
    """As above; uses module list explicitly to avoid nn.Module attribute issues."""
    modules = list(model.net)
    # output dimension K = out_features of the last linear layer
    K = None
    for m in reversed(modules):
        if isinstance(m, torch.nn.Linear):
            K = m.weight.shape[0]
            break
    glo = torch.zeros(K)
    ghi = torch.zeros(K)
    glo[k] = 1.0
    ghi[k] = 1.0
    linear_idx = 0
    # map: ReLU module index -> preact bounds of the preceding linear layer
    relu_preact = {}
    preacts = [op[1] for op in ops if op[0] == 'linear']
    lin_pos = 0
    for i, m in enumerate(modules):
        if isinstance(m, torch.nn.Linear):
            lin_pos += 1
        elif isinstance(m, torch.nn.ReLU):
            relu_preact[i] = preacts[lin_pos - 1]
    # reverse pass: modules[-1] is the last linear (output logits)
    for idx in range(len(modules) - 1, -1, -1):
        mod = modules[idx]
        if isinstance(mod, torch.nn.Linear):
            W = mod.weight
            Wp = torch.clamp(W, min=0.0)
            Wn = torch.clamp(W, max=0.0)
            glo_new = Wp.t() @ glo + Wn.t() @ ghi
            ghi_new = Wp.t() @ ghi + Wn.t() @ glo
            glo, ghi = _outward(glo_new, ghi_new)
        elif isinstance(mod, torch.nn.ReLU):
            pl, ph = relu_preact[idx]
            d_lo = (pl >= 0).float()
            d_hi = (~(ph <= 0)).float()
            glo, ghi = imul(glo, ghi, d_lo, d_hi)
    return glo, ghi


def margin_interval(logits_lo, logits_hi, c):
    """Interval enclosure of margin g = f_c - max_{j != c} f_j."""
    others = [j for j in range(logits_lo.shape[0]) if j != c]
    m_lo, m_hi = _outward(logits_lo[c] - torch.max(logits_hi[others]),
                           logits_hi[c] - torch.max(logits_lo[others]))
    return m_lo, m_hi


def margin_at(model, x, c):
    with torch.no_grad():
        logits = model(x.unsqueeze(0)).squeeze(0)
    others = [j for j in range(logits.shape[0]) if j != c]
    return float((logits[c] - torch.max(logits[others])).item())


def gradient_margin_hull(model, x_lo, x_hi, c, ops):
    """Sound interval enclosure J(X) of the gradient of the margin function
    over the box, obtained as the hull of pairwise-margin gradient intervals."""
    K = 0
    for m in model.net:
        if isinstance(m, torch.nn.Linear):
            K = m.weight.shape[0]
    d = x_lo.shape[0]
    hull_lo = torch.full((d,), float('inf'))
    hull_hi = torch.full((d,), float('-inf'))
    g_c_lo, g_c_hi = gradient_interval_logit(model, x_lo, x_hi, c, ops)
    for j in range(K):
        if j == c:
            continue
        g_j_lo, g_j_hi = gradient_interval_logit(model, x_lo, x_hi, j, ops)
        # gradient of f_c - f_j
        dg_lo, dg_hi = _outward(g_c_lo - g_j_hi, g_c_hi - g_j_lo)
        hull_lo = torch.minimum(hull_lo, dg_lo)
        hull_hi = torch.maximum(hull_hi, dg_hi)
    return hull_lo, hull_hi


def point_gradient_margin(model, x, c):
    x = x.detach().clone().requires_grad_(True)
    logits = model(x.unsqueeze(0)).squeeze(0)
    others = [j for j in range(logits.shape[0]) if j != c]
    g = logits[c] - torch.max(logits[others])
    g.backward()
    return x.grad.detach().clone()


# ---------------------------------------------------------------------------
# Interval-Krawczyk operator and certificates
# ---------------------------------------------------------------------------

def moore_penrose(g_vec, eps=1e-12):
    """Moore-Penrose pseudo-inverse of a row vector (returned as d-vector)."""
    n = g_vec.norm().item()
    if n < eps:
        return torch.zeros_like(g_vec)
    return g_vec / (n * n)


def krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi):
    """K(X) = m - C g(m) + (I - C J(X))(X - m), for the scalar equation
    g(x) = 0 in d unknowns (C is the d x 1 Moore-Penrose preconditioner)."""
    d = X_lo.shape[0]
    # C J(X): d x d interval matrix with entry (i,j) = C_i * J_j
    CJ_lo = torch.zeros(d, d)
    CJ_hi = torch.zeros(d, d)
    for i in range(d):
        ci = C[i].item()
        if ci >= 0:
            CJ_lo[i], CJ_hi[i] = _outward(ci * J_lo, ci * J_hi)
        else:
            CJ_lo[i], CJ_hi[i] = _outward(ci * J_hi, ci * J_lo)
    # I - C J
    M_lo = -CJ_lo
    M_hi = -CJ_hi
    diag_lo, diag_hi = _outward(1.0 - CJ_hi.diag(), 1.0 - CJ_lo.diag())
    M_lo[torch.arange(d), torch.arange(d)] = diag_lo
    M_hi[torch.arange(d), torch.arange(d)] = diag_hi
    # (X - m)
    xm_lo, xm_hi = _outward(X_lo - m, X_hi - m)
    prod_lo, prod_hi = interval_matvec(M_lo, M_hi, xm_lo, xm_hi)
    K_lo, K_hi = _outward(m - C * g_m + prod_lo, m - C * g_m + prod_hi)
    return K_lo, K_hi


# ---------------------------------------------------------------------------
# Verification methods
# ---------------------------------------------------------------------------

def verify_ibp(model, x0, c, eps):
    """IBP range test: robust iff margin lower bound > 0."""
    X_lo, X_hi = _outward(x0 - eps, x0 + eps)
    logits_lo, logits_hi, _ = forward_bounds(model, X_lo, X_hi)
    m_lo, _ = margin_interval(logits_lo, logits_hi, c)
    return 'robust' if m_lo > 0 else 'undecided'


def verify_krawczyk_classical(model, x0, c, eps):
    """One-shot classical interval-Krawczyk test on the box."""
    X_lo, X_hi = _outward(x0 - eps, x0 + eps)
    ops = forward_bounds(model, X_lo, X_hi)[2]
    m = (X_lo + X_hi) / 2.0
    g_m = margin_at(model, m, c)
    if g_m <= 0:
        return 'exist'  # the midpoint itself violates robustness
    J_lo, J_hi = gradient_margin_hull(model, X_lo, X_hi, c, ops)
    J_mid = (J_lo + J_hi) / 2.0
    C = moore_penrose(J_mid)
    K_lo, K_hi = krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
    if strict_inclusion(X_lo, X_hi, K_lo, K_hi):
        return 'exist'
    if empty_intersection(X_lo, X_hi, K_lo, K_hi):
        return 'robust'
    return 'undecided'


def extract_root(model, X_lo, X_hi, c, max_iter=40, tol=1e-9):
    """Refine a box by classical contraction after strict inclusion and return
    the midpoint as a high-accuracy root (adversarial example) candidate."""
    for _ in range(max_iter):
        _, _, ops = forward_bounds(model, X_lo, X_hi)
        m = (X_lo + X_hi) / 2.0
        g_m = margin_at(model, m, c)
        J_lo, J_hi = gradient_margin_hull(model, X_lo, X_hi, c, ops)
        J_mid = (J_lo + J_hi) / 2.0
        C = moore_penrose(J_mid)
        K_lo, K_hi = krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
        cand_lo = torch.maximum(X_lo, K_lo)
        cand_hi = torch.minimum(X_hi, K_hi)
        if (not empty_intersection(X_lo, X_hi, cand_lo, cand_hi)
                and strict_shrink(X_lo, X_hi, cand_lo, cand_hi)):
            X_lo, X_hi = cand_lo, cand_hi
        else:
            break
    m = (X_lo + X_hi) / 2.0
    return m, margin_at(model, m, c)


def verify_safe2(model, x0, c, eps, alpha_min=0.2, alpha_factor=0.8,
                 max_iter_r=15, max_iter_e=15, tol=1e-6):
    """SAFE2 gated relaxed protocol with sound dual certificates.

    Phase R (root-preserving classical chain): the box is contracted only with
    the classical operator K_1, which preserves every root of the original box.
    Robustness certificates ('robust') are issued only in this phase, via the
    range test (margin lower bound > 0) or the classical nonexistence test
    (X cap K_1(X) = empty), and are therefore sound w.r.t. the original box.

    Phase E (gated relaxed existence search): relaxed operators K_alpha are
    used only to accelerate progress toward a classical strict inclusion
    K_1(X) subset X, which certifies that a root (i.e. an adversarial example)
    exists in the box. Relaxed steps never support robustness claims.

    Returns (certificate, iterations, width_history, root, root_margin);
    root/root_margin are set only for 'exist' certificates.
    """
    X_lo, X_hi = _outward(x0 - eps, x0 + eps)
    width_hist = []
    root = None
    root_margin = None

    # ---- Phase R: root-preserving classical chain -------------------------
    it = 0
    for it in range(max_iter_r):
        logits_lo, logits_hi, ops = forward_bounds(model, X_lo, X_hi)
        m_lo, m_hi = margin_interval(logits_lo, logits_hi, c)
        m = (X_lo + X_hi) / 2.0
        g_m = margin_at(model, m, c)
        if g_m <= 0:
            return 'exist', it, width_hist, m, g_m
        if m_lo > 0:
            return 'robust', it, width_hist, root, root_margin
        J_lo, J_hi = gradient_margin_hull(model, X_lo, X_hi, c, ops)
        J_mid = (J_lo + J_hi) / 2.0
        C = moore_penrose(J_mid)
        K1_lo, K1_hi = krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
        if strict_inclusion(X_lo, X_hi, K1_lo, K1_hi):
            root, root_margin = extract_root(model, X_lo, X_hi, c)
            return 'exist', it, width_hist, root, root_margin
        if empty_intersection(X_lo, X_hi, K1_lo, K1_hi):
            return 'robust', it, width_hist, root, root_margin
        cand_lo = torch.maximum(X_lo, K1_lo)
        cand_hi = torch.minimum(X_hi, K1_hi)
        if strict_shrink(X_lo, X_hi, cand_lo, cand_hi):
            X_lo, X_hi = cand_lo, cand_hi
            width_hist.append(box_width(X_lo, X_hi))
            if box_width(X_lo, X_hi) < tol:
                return 'undecided', it, width_hist, root, root_margin
        else:
            break

    # ---- Phase E: gated relaxed existence search --------------------------
    for it_e in range(max_iter_e):
        it = max_iter_r + it_e
        _, _, ops = forward_bounds(model, X_lo, X_hi)
        m = (X_lo + X_hi) / 2.0
        g_m = margin_at(model, m, c)
        if g_m <= 0:
            return 'exist', it, width_hist, m, g_m
        J_lo, J_hi = gradient_margin_hull(model, X_lo, X_hi, c, ops)
        J_mid = (J_lo + J_hi) / 2.0
        C = moore_penrose(J_mid)
        K1_lo, K1_hi = krawczyk_image(m, g_m, J_lo, J_hi, C, X_lo, X_hi)
        if strict_inclusion(X_lo, X_hi, K1_lo, K1_hi):
            root, root_margin = extract_root(model, X_lo, X_hi, c)
            return 'exist', it, width_hist, root, root_margin
        if empty_intersection(X_lo, X_hi, K1_lo, K1_hi):
            break  # no root in the current box; no existence cert possible
        # gated relaxed contraction
        alpha = 1.0
        accepted = False
        for _try in range(16):  # bounded; alpha >= alpha_min only via tries
            Ja_lo = J_mid + alpha * (J_lo - J_mid)
            Ja_hi = J_mid + alpha * (J_hi - J_mid)
            Ka_lo, Ka_hi = krawczyk_image(m, g_m, Ja_lo, Ja_hi, C, X_lo, X_hi)
            cand_lo = torch.maximum(X_lo, Ka_lo)
            cand_hi = torch.minimum(X_hi, Ka_hi)
            if strict_shrink(X_lo, X_hi, cand_lo, cand_hi):
                X_lo, X_hi = cand_lo, cand_hi
                accepted = True
                break
            alpha *= alpha_factor
            if alpha < alpha_min - 1e-9:
                break
        if not accepted:
            cand_lo = torch.maximum(X_lo, K1_lo)
            cand_hi = torch.minimum(X_hi, K1_hi)
            if strict_shrink(X_lo, X_hi, cand_lo, cand_hi):
                X_lo, X_hi = cand_lo, cand_hi
            else:
                break
        width_hist.append(box_width(X_lo, X_hi))
        if box_width(X_lo, X_hi) < tol:
            break
    return 'undecided', it, width_hist, root, root_margin


# ---------------------------------------------------------------------------
# PGD reference attack (L_inf)
# ---------------------------------------------------------------------------

def pgd_attack(model, x0, c, eps, steps=20, restarts=5, step_size=0.05):
    """Multi-restart PGD; returns True if an adversarial example is found."""
    x0 = x0.detach().clone()
    for _ in range(restarts):
        noise = (torch.rand_like(x0) * 2 - 1) * eps
        x = torch.clamp(x0 + noise, 0.0, 1.0)
        x = x.requires_grad_(True)
        for _ in range(steps):
            logits = model(x.unsqueeze(0)).squeeze(0)
            others = [j for j in range(logits.shape[0]) if j != c]
            loss = logits[c] - torch.max(logits[others])
            loss.backward()
            with torch.no_grad():
                x = x + step_size * torch.sign(x.grad)
                x = torch.clamp(x, x0 - eps, x0 + eps)
                x = torch.clamp(x, 0.0, 1.0)
            x = x.detach().requires_grad_(True)
        with torch.no_grad():
            logits = model(x.unsqueeze(0)).squeeze(0)
            pred = int(torch.argmax(logits).item())
            if pred != c:
                return True
    return False
