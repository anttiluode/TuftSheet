"""The two-context world, its ideal observer, and one body.

A row of N = 24 columns. Columns 0-11 are the LEFT region, 12-23 the RIGHT. Each region has its own hidden
rule, +-1, and each rule flips on its own, uncued, with probability HAZARD every trial (mean block 30).
(A first version used 20-40-trial blocks; the two rules then stayed in or out of step for long stretches, and
one global g learned to exploit that anti-correlation. A constant hazard makes the regions truly independent.) Each trial cues one column,
uniformly: it sees A = +-1, then the outcome B = rule(region) * A, flipped with probability 1 - Q.
Nothing in the machines knows the regions, where the boundary is, or when a rule flips.

The ideal observer knows all of that: one exact two-state filter per region, stepped every trial with the
hazard, updated only by trials in its own region.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sheet.sheet import Sheet, GlobalG

N, BOUND, Q = 24, 12, 0.9
HAZARD = 1 / 30
TRAIN, TEST = 4000, 600
MODES = ("learned", "uniform", "local", "oracle", "global")


def region(i):
    return 0 if i < BOUND else 1


def sequence(rng, n):
    rules = np.zeros((n, 2), int); r = [int(rng.choice([1, -1])) for _ in range(2)]
    cols, A, B = [], [], []
    for t in range(n):
        for k in range(2):
            if t > 0 and rng.random() < HAZARD:
                r[k] = -r[k]
        rules[t] = r
        i = int(rng.integers(0, N)); a = int(rng.choice([1, -1])); b = r[region(i)] * a * (1 if rng.random() < Q else -1)
        cols.append(i); A.append(a); B.append(b)
    return np.array(cols), np.array(A), np.array(B), rules


def ideal(cols, A, B):
    """E[B * A] (along A) before each trial: (2Q-1)(2P(rule of its region = +1) - 1). Exact two-state filter."""
    P = [0.5, 0.5]; out = []
    for i, a, b in zip(cols, A, B):
        P = [p * (1 - HAZARD) + (1 - p) * HAZARD for p in P]
        k = region(i); out.append((2 * Q - 1) * (2 * P[k] - 1))
        lp, lm = (Q if a * 1 == b else 1 - Q), (Q if a * -1 == b else 1 - Q)
        P[k] = P[k] * lp / (P[k] * lp + (1 - P[k]) * lm)
    return np.array(out)
def make(mode, **kw):
    if mode == "global":
        return GlobalG(N)
    return Sheet(N, mode=mode, boundary=BOUND, **kw)


def body(seed, modes=MODES, sheet_kw=None):
    rng = np.random.default_rng(seed)
    tr = sequence(rng, TRAIN); te = sequence(rng, TEST)
    out = {"seed": seed, "cols": te[0].tolist(), "A": te[1].tolist(), "B": te[2].tolist(), "rules": te[3].tolist(),
           "ideal": ideal(*te[:3]).tolist()}
    for m in modes:
        M = make(m, **(sheet_kw or {}) if m != "global" else {})
        for i, a, b in zip(*tr[:3]):
            M.live(int(i), int(a), int(b), learn=True)
        pred, cut = [], []
        for i, a, b in zip(*te[:3]):
            pred.append(M.predict(int(i), int(a)) * a); cut.append(M.predict(int(i), int(a), apical=False) * a)
            M.live(int(i), int(a), int(b), learn=False)
        out[m] = pred; out[m + "_cut"] = cut
        if m != "global":
            out[m + "_c"] = M.c.tolist()
        out[m + "_V"] = M.V.tolist(); out[m + "_W"] = M.W.tolist()
    return out
