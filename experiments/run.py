"""Pre-registered batch (PROTOCOL.md): 32 bodies, seeds 600001-600032. About a minute on two cores."""
import hashlib, json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE))
import world as Wd
SEEDS = list(range(600001, 600033))
if __name__ == "__main__":
    files = ["PROTOCOL.md", "experiments/run.py", "experiments/world.py", "experiments/metrics.py", "sheet/sheet.py"]
    rec = {"label": "TuftSheet batch (pre-registered)", "seeds": SEEDS,
           "hashes": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}, "results": []}
    t0 = time.time()
    with ProcessPoolExecutor(2) as p:
        rec["results"] = list(p.map(Wd.body, SEEDS))
    rec["elapsed_seconds"] = time.time() - t0
    (HERE / "results.json").write_text(json.dumps(rec)); print("done %.0fs" % rec["elapsed_seconds"])
