import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpolib import dpobatch, dpoloss, score, template

ap = argparse.ArgumentParser()
ap.add_argument("--sft", required=True)
ap.add_argument("--train", required=True)
ap.add_argument("--val", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--lr", type=float, default=1e-6)
ap.add_argument("--accum", type=int, default=32)
ap.add_argument("--n", type=int, default=15000)
ap.add_argument("--valn", type=int, default=1000)
ap.add_argument("--every", type=int, default=100)
ap.add_argument("--limit", type=int, default=512)
args = ap.parse_args()

random.seed(42)
torch.manual_seed(42)
os.makedirs(args.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(args.sft)
policy = AutoModelForCausalLM.from_pretrained(args.sft, torch_dtype=torch.float32).to("cuda:0")
policy.config.use_cache = False
rdev = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
ref = AutoModelForCausalLM.from_pretrained(args.sft, torch_dtype=torch.float16).to(rdev).eval()
for p in ref.parameters():
    p.requires_grad_(False)

train = json.load(open(args.train))[: args.n]
val = json.load(open(args.val))[: args.valn]


def pair(r):
    return dpobatch(tok, [template(r["q"])], [r["good"]], [r["bad"]], args.limit)


@torch.no_grad()
def refs(rows):
    out = []
    for r in rows:
        b = pair(r)
        out.append((score(ref, b["chosen"]).item(), score(ref, b["rejected"]).item()))
    return out


cache = refs(val)
opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
logf = open(os.path.join(args.out, "metrics.jsonl"), "a")


def log(row):
    logf.write(json.dumps(row) + "\n")
    logf.flush()
    print(row, flush=True)


@torch.no_grad()
def check():
    policy.eval()
    losses = []
    gaps = []
    wins = 0
    for r, (cw, cl) in zip(val, cache):
        b = pair(r)
        with torch.autocast("cuda", dtype=torch.float16):
            pc = score(policy, b["chosen"])
            pr = score(policy, b["rejected"])
        rc = torch.tensor([cw], device=pc.device)
        rr = torch.tensor([cl], device=pc.device)
        losses.append(dpoloss(pc, pr, rc, rr, args.beta).item())
        gw = args.beta * (pc - rc)
        gl = args.beta * (pr - rr)
        wins += int((gw > gl).item())
        gaps.append((gw - gl).item())
    policy.train()
    return sum(losses) / len(losses), wins / len(val), sum(gaps) / len(gaps)


def keep(name):
    path = os.path.join(args.out, name)
    policy.save_pretrained(path)
    tok.save_pretrained(path)


steps = len(train) // args.accum
order = list(range(len(train)))
random.shuffle(order)
best = (-1.0, -1e9)
t0 = time.time()

vl, va, vm = check()
log({"kind": "val", "step": 0, "loss": vl, "acc": va, "margin": vm})

policy.train()
for step in range(steps):
    tot = 0.0
    hit = 0
    for k in range(args.accum):
        r = train[order[step * args.accum + k]]
        b = pair(r)
        with torch.no_grad():
            rc = score(ref, b["chosen"])
            rr = score(ref, b["rejected"])
        with torch.autocast("cuda", dtype=torch.float16):
            pc = score(policy, b["chosen"])
            pr = score(policy, b["rejected"])
        rc = rc.to(pc.device)
        rr = rr.to(pc.device)
        loss = dpoloss(pc, pr, rc, rr, args.beta)
        scaler.scale(loss / args.accum).backward()
        tot += loss.item() / args.accum
        hit += int(((pc - rc) > (pr - rr)).item())
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    if (step + 1) % 5 == 0 or step == 0:
        log({"kind": "train", "step": step + 1, "loss": tot, "acc": hit / args.accum, "mins": (time.time() - t0) / 60})
    if (step + 1) % args.every == 0 or step + 1 == steps:
        vl, va, vm = check()
        log({"kind": "val", "step": step + 1, "loss": vl, "acc": va, "margin": vm})
        if (va, vm) > best:
            best = (va, vm)
            keep("best")
keep("final")
print("done, best val acc and margin", best)
