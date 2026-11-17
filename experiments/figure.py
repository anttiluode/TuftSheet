"""experiments/tuftsheet.png: the sheet of one body over time, and the batch."""
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import world as Wd
from metrics import measures
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
COL = {"ideal": "#1f2328", "oracle": "#5b6cc4", "uniform": "#1a7f64", "learned": "#7fbfa9", "global": "#cf6a2e", "local": "#afb8c1"}

# one body, re-run with the sheet recorded (same seed and code as the batch)
seed = 600001; rng = np.random.default_rng(seed)
tr = Wd.sequence(rng, Wd.TRAIN); te = Wd.sequence(rng, Wd.TEST)
M = Wd.make("uniform")
for i, a, b in zip(*tr[:3]):
    M.live(int(i), int(a), int(b))
snap = []
for i, a, b in zip(*te[:3]):
    snap.append(np.tanh(M.phi) * np.sign(np.mean(M.V))); M.live(int(i), int(a), int(b), learn=False)
snap = np.array(snap); rules = te[3]

d = json.loads((HERE / "results.json").read_text())["results"]; ms = [measures(b) for b in d]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(13, 8)); gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
a = fig.add_subplot(gs[0, :]); T = 300
a.imshow(snap[:T].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", extent=[0, T, -0.5, Wd.N - 0.5])
for k, (lo, hi) in enumerate(((-0.5, Wd.BOUND - 0.5), (Wd.BOUND - 0.5, Wd.N - 0.5))):
    for t in range(1, T):
        if rules[t, k] != rules[t - 1, k]:
            a.plot([t, t], [lo, hi], color=INK, lw=1.2)
a.axhline(Wd.BOUND - 0.5, color=INK, lw=0.8, ls="--")
a.scatter(np.arange(T), te[0][:T], s=4, color=INK, alpha=.35, lw=0)
a.set_xlabel("test trial (weights frozen; black ticks: a region's hidden rule flips; dots: the cued column)")
a.set_ylabel("column (dashed: the boundary\nthe sheet is never told)")
a.set_title("The sheet of one body: each region holds its own context; a flip spreads as a front, mostly within its own region", loc="left", color=INK, fontsize=10)
rng2 = np.random.default_rng(0)
def strip(ax, keys, metric, ylab, title, ylim=(0, 1.02), chance=True):
    for j, k in enumerate(keys):
        v = np.array([m[k + metric] for m in ms]); ax.scatter(j + rng2.uniform(-.15, .15, len(v)), v, s=7, color=COL[k], alpha=.5, lw=0)
        ax.plot([j - .3, j + .3], [np.nanmedian(v)] * 2, color=COL[k], lw=2.5)
    if chance:
        ax.axhline(.5, color=INK, lw=.8, ls="--")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(["global g" if k == "global" else ("no sheet" if k == "local" else k) for k in keys], rotation=25)
    ax.set_ylim(*ylim); ax.set_ylabel(ylab); ax.set_title(title, loc="left", color=INK, fontsize=10)
strip(fig.add_subplot(gs[1, 0]), ["ideal", "oracle", "uniform", "learned", "global", "local"], "_acc", "sign accuracy",
      "Two contexts at once", ylim=(0.4, 1.0))
a = fig.add_subplot(gs[1, 1])
for j, k in enumerate(["ideal", "uniform", "local"]):
    for o, x, lab in ((-.17, "_transfer_near", "near"), (.17, "_transfer_far", "far")):
        v = np.array([m[k + x] for m in ms]); a.bar(j + o, np.nanmedian(v), width=.3, color=COL[k], alpha=1 if lab == "near" else .45)
        a.text(j + o, np.nanmedian(v) + .02, lab, ha="center", fontsize=7.5, color=MUTED)
a.axhline(.5, color=INK, lw=.8, ls="--"); a.set_xticks(range(3)); a.set_xticklabels(["ideal", "uniform sheet", "no sheet"])
a.set_ylim(0, 1.02); a.set_ylabel("first-trial accuracy after a switch")
a.set_title("Evidence travels: near (≤ 3 columns) vs far (≥ 6)", loc="left", color=INK, fontsize=10)
a = fig.add_subplot(gs[1, 2])
C = np.array([b["learned_c"] for b in d]); x = np.arange(C.shape[1]) + 0.5
a.plot(x, C.T, color=COL["learned"], lw=.6, alpha=.35); a.plot(x, np.median(C, 0), color=COL["uniform"], lw=2.2, label="median")
a.axvline(Wd.BOUND - 0.5, color=INK, lw=.8, ls="--"); a.set_ylim(0, 1.02); a.set_xlabel("link between column i and i+1")
a.set_ylabel("learned link strength"); a.legend(frameon=False, fontsize=8, loc="lower left")
a.set_title("Learning the boundary: a dip, not a cut", loc="left", color=INK, fontsize=10)
fig.text(.995, .005, "32 bodies · dots: bodies · bars: medians", ha="right", fontsize=8, color=MUTED)
fig.tight_layout(); fig.savefig(HERE / "tuftsheet.png", dpi=150); print("ok")
