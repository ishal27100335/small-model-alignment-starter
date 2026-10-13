import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

kind, path, out = sys.argv[1:4]


def smooth(v, k=10):
    res = []
    for i in range(len(v)):
        w = v[max(0, i - k + 1) : i + 1]
        res.append(sum(w) / len(w))
    return res


if kind == "lora":
    hist = [h for h in json.load(open(path)) if "loss" in h]
    xs = [h["step"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(xs, [h["loss"] for h in hist])
    ax[0].set_title("LoRA DPO train loss")
    ax[1].plot(xs, [h.get("rewards/accuracies", 0) for h in hist])
    ax[1].set_title("reward accuracy (train)")
    ax[2].plot(xs, [h.get("rewards/margins", 0) for h in hist])
    ax[2].set_title("reward margin (train)")
    for a in ax:
        a.set_xlabel("optimizer step")
else:
    rows = [json.loads(l) for l in open(path)]
    tr = [r for r in rows if r["kind"] == "train"]
    va = [r for r in rows if r["kind"] == "val"]
    if kind == "sft":
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.plot([r["step"] for r in tr], [r["loss"] for r in tr], alpha=0.3, label="train raw")
        ax.plot([r["step"] for r in tr], smooth([r["loss"] for r in tr]), label="train smoothed (10 logs)")
        ax.plot([r["step"] for r in va], [r["loss"] for r in va], "o-", label="validation")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("token NLL")
        ax.legend()
    else:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot([r["step"] for r in tr], smooth([r["loss"] for r in tr]), label="train smoothed")
        ax[0].plot([r["step"] for r in va], [r["loss"] for r in va], "o-", label="validation")
        ax[0].set_title("DPO loss")
        ax[0].legend()
        ax[1].plot([r["step"] for r in va], [r["acc"] for r in va], "o-")
        ax[1].set_title("validation reward accuracy")
        ax[2].plot([r["step"] for r in va], [r["margin"] for r in va], "o-")
        ax[2].set_title("validation reward margin")
        for a in ax:
            a.set_xlabel("optimizer step")
plt.tight_layout()
plt.savefig(out, dpi=150)
print("saved", out)
