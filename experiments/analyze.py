"""Score against PROTOCOL.md.   python experiments/analyze.py [results.json]"""
import json, math, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import world as Wd
from metrics import measures


def summarize(path):
    ms = [measures(b) for b in json.loads(pathlib.Path(path).read_text())["results"]]
    n = len(ms); n90, n75 = math.ceil(0.9 * n), math.ceil(0.75 * n)
    col = lambda k: np.array([m[k] for m in ms]); med = lambda k: float(np.nanmedian(col(k)))
    R = {}
    bg = int((col("uniform_acc") > col("global_acc")).sum()); bl = int((col("uniform_acc") > col("local_acc")).sum())
    R["T1_two_contexts"] = {"uniform_acc": med("uniform_acc"), "global_acc": med("global_acc"), "local_acc": med("local_acc"),
                            "ideal_acc": med("ideal_acc"), "beats_global": bg, "beats_local": bl,
                            "pass": bool(med("uniform_acc") >= 0.75 and bg >= n90 and bl >= n90)}
    R["T2_sideways_transfer"] = {"uniform_near": med("uniform_transfer_near"), "local_near": med("local_transfer_near"),
                                 "ideal_near": med("ideal_transfer_near"),
                                 "pass": bool(med("uniform_transfer_near") >= 0.70 and med("local_transfer_near") <= 0.50)}
    ab = int((col("uniform_leak") > col("global_leak")).sum())
    R["T3_boundary_holds"] = {"uniform": med("uniform_leak"), "global": med("global_leak"), "ideal": med("ideal_leak"),
                              "above_global": ab, "pass": bool(med("uniform_leak") >= 0.85 and ab >= n75)}
    gap = (col("uniform_transfer_near") - col("uniform_transfer_far")) - (col("ideal_transfer_near") - col("ideal_transfer_far"))
    R["T4_evidence_travels"] = {"sheet_gap": float(np.nanmedian(col("uniform_transfer_near") - col("uniform_transfer_far"))),
                                "ideal_gap": float(np.nanmedian(col("ideal_transfer_near") - col("ideal_transfer_far"))),
                                "median_extra_gap": float(np.nanmedian(gap)), "pass": bool(np.nanmedian(gap) > 0.05)}
    R["T5_cut"] = {"uniform_cut_acc": med("uniform_cut_acc"), "pass": bool(0.40 <= med("uniform_cut_acc") <= 0.60)}
    below = int((col("boundary_link") < col("within_link")).sum())
    R["T6_boundary_learned"] = {"below_within": below, "boundary_rank_median": med("boundary_rank"),
                                "weakest_at_boundary": int(col("weakest_link_at_boundary").sum()), "pass": bool(below >= n75)}
    rep = {m: {x: med(m + x) for x in ("_acc", "_transfer", "_transfer_near", "_transfer_far", "_leak", "_mse")}
           for m in ("ideal",) + Wd.MODES}
    rep["learned_beats_uniform"] = int((col("learned_acc") > col("uniform_acc")).sum())
    rep["oracle_beats_uniform"] = int((col("oracle_acc") > col("uniform_acc")).sum())
    return {"n": n, "rules": R, "report": rep, "all_rules_pass": all(v["pass"] for v in R.values())}


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.json"
    S = summarize(path)
    (HERE / ("summary.json" if path.name == "results.json" else "dev_summary.json")).write_text(json.dumps(S, indent=1))
    for k, v in S["rules"].items():
        print("%-22s %s %s" % (k, "PASS" if v["pass"] else "FAIL", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "pass"}))
    print("all rules pass:", S["all_rules_pass"])
