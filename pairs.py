import argparse
import json
import random

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True)
ap.add_argument("--b", required=True)
ap.add_argument("--an", required=True)
ap.add_argument("--bn", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

rng = random.Random(args.seed)
first = {r["id"]: r for r in json.load(open(args.a))}
second = {r["id"]: r for r in json.load(open(args.b))}
with open(args.out, "w") as f:
    for key in sorted(first):
        x = first[key]
        y = second[key]
        if rng.random() < 0.5:
            row = {"id": key, "instruction": x["instruction"], "a": x["output"], "b": y["output"], "meta": {"A": args.an, "B": args.bn}}
        else:
            row = {"id": key, "instruction": x["instruction"], "a": y["output"], "b": x["output"], "meta": {"A": args.bn, "B": args.an}}
        f.write(json.dumps(row) + "\n")
print("wrote", len(first), "pairs")
