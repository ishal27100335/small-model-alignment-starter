import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpolib import template

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tok", default=None)
ap.add_argument("--lora", default=None)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--chat", action="store_true")
ap.add_argument("--n", type=int, default=0)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--new", type=int, default=256)
ap.add_argument("--pen", type=float, default=1.1)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.tok or args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda:0")
if args.lora:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.lora)
model.eval()

rows = json.load(open(args.prompts))
if args.n:
    rows = rows[: args.n]


def wrap(q):
    if args.chat:
        msg = [{"role": "user", "content": q}]
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return template(q)


order = sorted(range(len(rows)), key=lambda i: len(rows[i]["instruction"]))
outs = {}
for s in range(0, len(order), args.bs):
    ids = order[s : s + args.bs]
    enc = tok([wrap(rows[i]["instruction"]) for i in ids], return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=args.new,
            do_sample=False,
            repetition_penalty=args.pen,
            pad_token_id=tok.pad_token_id,
        )
    texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    for i, t in zip(ids, texts):
        outs[i] = t.split("### Instruction:")[0].strip()
    print(s + len(ids), "/", len(rows), flush=True)

res = []
for i, r in enumerate(rows):
    res.append(
        {
            "id": r["id"],
            "instruction": r["instruction"],
            "output": outs[i],
            "generator": args.name,
            "dataset": r.get("dataset", "unknown"),
        }
    )
json.dump(res, open(args.out, "w"), indent=1)
for r in res[:3]:
    print("-----", r["instruction"], "\n", r["output"][:600])
