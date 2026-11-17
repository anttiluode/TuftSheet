"""TuftSheet: relational columns under a shared Layer-1 sheet.

Each column i is the relational neuron of WaveCortex, in its scalar form (WaveCortex showed the scalar copy
is the more accurate one):
    prediction_i = A * (W_i + g_i * V_i),     g_i = tanh(phi_i)
  A = +-1 is the cue the column sees; W_i (basal) and V_i (apical-gated) are slow weights; g_i is what the
  column's tuft reads from the sheet directly above it.

The sheet (Layer 1) is a field phi over the row of columns, read as log-odds of the hidden rule:
  ping:    after a trial at column i, its surprise is pushed up into the sheet at its own place,
               phi_i <- phi_i - eta * r_i * A * V_i / (V_i^2 + kappa),    r_i = prediction_i - outcome
           (the normalized surprise step of ApicalRuleReversalMachine, now local)
  spread:  between trials, K ticks of   phi <- phi + dt * [ sum_j c_ij (phi_j - phi_i) - leak * phi ]
           neighbours share evidence sideways; the leak lets old evidence fade (rules can change)
  coupling c_ij between neighbours is LEARNED, slowly: each column keeps a trace of its recent pushes; when
           two neighbours' traces agree in sign the link strengthens, when they disagree it weakens.
           Nothing tells the sheet where one context ends and another begins.
Weights learn three-factor, as before:  W_i <- W_i - k r_i A,   V_i <- V_i - k g_i r_i A.
"""
from __future__ import annotations
import numpy as np


class Sheet:
    def __init__(self, n, mode="learned", D=10.0, leak=0.02, K=25, dt=0.04, eta=8.0, kappa=0.1,
                 lr=0.03, eps_c=0.01, trace_decay=0.98, phi0=2.0, boundary=None):
        # defaults chosen on development seeds 981-986 only (PROTOCOL.md)
        self.n, self.mode = n, mode
        self.D, self.leak, self.K, self.dt, self.eta, self.kappa, self.lr = D, leak, K, dt, eta, kappa, lr
        self.eps_c, self.td = eps_c, trace_decay
        self.W = np.zeros(n); self.V = np.zeros(n)
        self.phi = np.full(n, phi0)                       # starts at one context: breaks the sign symmetry
        self.c = np.ones(n - 1)                           # link between column i and i+1
        if mode == "oracle":                              # the true boundary given (upper bound for a sheet)
            self.c[boundary - 1] = 0.0
        if mode == "local":                               # no sheet: every column alone
            self.c[:] = 0.0
        self.trace = np.zeros(n)

    def g(self):
        return np.tanh(self.phi)

    def predict(self, i, A, apical=True):
        return A * (self.W[i] + (np.tanh(self.phi[i]) if apical else 0.0) * self.V[i])

    def live(self, i, A, B, learn=True):
        """One trial at column i: cue A, outcome B. Ping, (learn), then the sheet spreads."""
        gi = np.tanh(self.phi[i]); r = A * (self.W[i] + gi * self.V[i]) - B
        push = -self.eta * r * A * self.V[i] / (self.V[i] ** 2 + self.kappa)
        self.phi[i] += push
        if learn:
            self.W[i] -= self.lr * r * A
            self.V[i] -= self.lr * gi * r * A
        self.trace *= self.td; self.trace[i] += push
        if self.mode == "learned" and learn:
            agree = self.trace[:-1] * self.trace[1:]
            self.c = np.clip(self.c + self.eps_c * np.tanh(agree), 0.0, 1.0)
        for _ in range(self.K):
            lap = np.zeros(self.n)
            f = self.c * (self.phi[1:] - self.phi[:-1])
            lap[:-1] += f; lap[1:] -= f
            self.phi += self.dt * (self.D * lap - self.leak * self.phi)


class GlobalG:
    """ApicalRuleReversalMachine's scalar circuit: one g for every column (no sheet, no locality)."""
    def __init__(self, n, eta=0.5, kappa=0.1, lr=0.03):
        self.W = np.zeros(n); self.V = np.zeros(n); self.gv = 1.0; self.eta, self.kappa, self.lr = eta, kappa, lr

    def predict(self, i, A, apical=True):
        return A * (self.W[i] + (self.gv if apical else 0.0) * self.V[i])

    def live(self, i, A, B, learn=True):
        r = self.predict(i, A) - B
        gn = float(np.clip(self.gv - self.eta * r * A * self.V[i] / (self.V[i] ** 2 + self.kappa), -1, 1))
        if learn:
            self.W[i] -= self.lr * r * A; self.V[i] -= self.lr * self.gv * r * A
        self.gv = gn
