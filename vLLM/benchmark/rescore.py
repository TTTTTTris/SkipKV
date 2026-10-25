"""Re-score saved ``--save-outputs`` JSONL files with the current answer parser.

    python rescore.py skipkv_runs/math/*.jsonl

Prints old vs new accuracy per file; ``--write`` also rewrites each JSONL's
pred/ok fields and the matching summary JSON's accuracy.
"""
import argparse
import json
import os
import re

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval.py")).read()
_ns = {"re": re}
exec(_src[_src.index("_PATS = ["):_src.index("\n\n\n", _src.index("def to_num"))], _ns)
extract_pred, to_num = _ns["extract_pred"], _ns["to_num"]

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--write", action="store_true")
args = ap.parse_args()
for f in args.files:
    rows = [json.loads(l) for l in open(f)]
    old = sum(r["ok"] for r in rows)
    for r in rows:
        p = to_num(extract_pred(r["text"]))
        r["pred"], r["ok"] = p, r["gold"] is not None and p is not None and abs(r["gold"] - p) < 1e-4
    new = sum(r["ok"] for r in rows)
    print(f"{os.path.basename(f)[:-6]:28s} {old / len(rows):.3f} -> {new / len(rows):.3f}  (n={len(rows)})")
    if args.write:
        with open(f, "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        js = f[:-1]  # .jsonl -> .json
        if os.path.exists(js):
            d = json.load(open(js))
            for k in ("accuracy", "acc"):
                if k in d:
                    d[k] = new / len(rows)
            d["correct"] = new
            json.dump(d, open(js, "w"), indent=2)
