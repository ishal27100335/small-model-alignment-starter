import argparse
import json
import math
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpolib import collate, count, score, template

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/sft")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
ap.add_argument("--n", type=int, default=30000)
ap.add_argument("--valn", type=int, default=1000)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--every", type=int, default=150)
ap.add_argument("--limit", type=int, default=512)
args = ap.parse_args()

random.seed(42)
torch.manual_seed(42)
os.makedirs(args.out, exist_ok=True)
dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(dev)
model.config.use_cache = False
train = json.load(open("data/proc/sft_train.json"))[: args.n]
val = json.load(open("data/proc/sft_val.json"))[: args.valn]


def make(rows):
    return collate(tok, [(template(r["q"]), r["a"]) for r in rows], args.limit)


steps = len(train) // (args.bs * args.accum)
warm = max(1, int(0.03 * steps))
decay = [p for p in model.parameters() if p.ndim >= 2]
flat = [p for p in model.parameters() if p.ndim < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": flat, "weight_decay": 0.0}],
    lr=args.lr,
)


def shape(s):
    if s < warm:
        return (s + 1) / warm
    t = (s - warm) / max(1, steps - warm)
    return 0.5 * (1 + math.cos(math.pi * t))


sched = torch.optim.lr_scheduler.LambdaLR(opt, shape)
scaler = torch.amp.GradScaler("cuda")


@torch.no_grad()
def check():
    model.eval()
    tot = 0.0
    cnt = 0
    for i in range(0, len(val), 4):
        b = make(val[i : i + 4])
        with torch.autocast("cuda", dtype=torch.float16):
            lp = score(model, b)
        tot += -lp.sum().item()
        cnt += count(b).item()
    model.train()
    return tot / max(cnt, 1)


def keep(name):
    path = os.path.join(args.out, name)
    model.save_pretrained(path)
    tok.save_pretrained(path)


order = list(range(len(train)))
random.shuffle(order)
logf = open(os.path.join(args.out, "metrics.jsonl"), "a")


def log(row):
    logf.write(json.dumps(row) + "\n")
    logf.flush()
    print(row, flush=True)


best = 1e9
t0 = time.time()
model.train()
for step in range(steps):
    tot = 0.0
    for k in range(args.accum):
        at = (step * args.accum + k) * args.bs
        b = make([train[j] for j in order[at : at + args.bs]])
        with torch.autocast("cuda", dtype=torch.float16):
            lp = score(model, b)
        loss = -lp.sum() / count(b).to(lp.device)
        scaler.scale(loss / args.accum).backward()
        tot += loss.item() / args.accum
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    sched.step()
    if (step + 1) % 10 == 0 or step == 0:
        log({"kind": "train", "step": step + 1, "loss": tot, "lr": sched.get_last_lr()[0], "mins": (time.time() - t0) / 60})
    if (step + 1) % args.every == 0 or step + 1 == steps:
        vl = check()
        log({"kind": "val", "step": step + 1, "loss": vl})
        if vl < best:
            best = vl
            keep("best")
keep("final")
print("done, best val loss", best)
