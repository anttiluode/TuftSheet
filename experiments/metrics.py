"""Per-body measures. Predictions are read along A, so the right sign is the current rule of the column's region."""
import numpy as np
import world as Wd


def events(b):
    cols = np.array(b["cols"]); rules = np.array(b["rules"]); n = len(cols)
    reg = np.array([Wd.region(i) for i in cols]); truth = rules[np.arange(n), reg]
    # switch times per region
    sw = {k: [0] + [t for t in range(1, n) if rules[t, k] != rules[t - 1, k]] for k in (0, 1)}
    transfer, dist, leak = [], [], []
    for k in (0, 1):
        for s, e in zip(sw[k], sw[k][1:] + [n]):
            if s == 0:
                continue
            seen_cols = []
            for t in range(s, e):
                if reg[t] != k:
                    continue
                c = cols[t]
                if c not in seen_cols and seen_cols:
                    transfer.append(t); dist.append(min(abs(c - x) for x in seen_cols))
                if c not in seen_cols:
                    seen_cols.append(c)
        # leak: trials in the OTHER region within 10 trials after this region's switch, whose own rule did not change
        o = 1 - k
        for s in sw[k][1:]:
            for t in range(s, min(n, s + 10)):
                if reg[t] == o and all(rules[u, o] == rules[s, o] for u in range(max(0, s - 10), t + 1)):
                    leak.append(t)
    return reg, truth, transfer, np.array(dist), leak


def measures(b, modes=Wd.MODES):
    reg, truth, tr, dist, leak = events(b)
    rules = np.array(b["rules"]); differ = rules[:, 0] != rules[:, 1]
    idl = np.array(b["ideal"]); out = {"n_transfer": len(tr), "n_leak": len(leak)}
    for m in ("ideal",) + tuple(modes):
        x = np.array(b[m]); ok = np.sign(x) == truth
        out[m + "_acc"] = float(ok.mean()); out[m + "_acc_differ"] = float(ok[differ].mean())
        out[m + "_transfer"] = float(ok[tr].mean()) if tr else float("nan")
        out[m + "_transfer_near"] = float(ok[np.array(tr)[dist <= 3]].mean()) if (dist <= 3).any() else float("nan")
        out[m + "_transfer_far"] = float(ok[np.array(tr)[dist >= 6]].mean()) if (dist >= 6).any() else float("nan")
        out[m + "_leak"] = float(ok[leak].mean()) if leak else float("nan")
        out[m + "_mse"] = float(np.mean((x - idl) ** 2))
        if m != "ideal":
            out[m + "_cut_acc"] = float((np.sign(np.array(b[m + "_cut"])) == truth).mean())
    if "learned_c" in b:
        c = np.array(b["learned_c"]); bi = Wd.BOUND - 1
        out["boundary_link"] = float(c[bi]); out["within_link"] = float(np.median(np.delete(c, bi)))
        out["weakest_link_at_boundary"] = bool(int(np.argmin(c)) == bi)
        out["boundary_rank"] = float(np.mean(np.delete(c, bi) < c[bi]))   # 0: weakest of all; 0.5: typical
    return out
