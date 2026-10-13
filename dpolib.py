import torch
import torch.nn.functional as F


def template(q):
    return "### Instruction:\n" + q.strip() + "\n\n### Response:\n"


def encode(tok, q, a, limit):
    qids = tok.encode(q, add_special_tokens=False)
    aids = tok.encode(a, add_special_tokens=False)
    if tok.eos_token_id is not None:
        aids = aids + [tok.eos_token_id]
    ids = (qids + aids)[:limit]
    labs = ([-100] * len(qids) + aids)[:limit]
    return ids, labs


def collate(tok, pairs, limit):
    rows = [encode(tok, q, a, limit) for q, a in pairs]
    width = max(len(r[0]) for r in rows)
    padid = tok.pad_token_id
    if padid is None:
        padid = tok.eos_token_id or 0
    ids = torch.tensor([r[0] + [padid] * (width - len(r[0])) for r in rows])
    mask = torch.tensor([[1] * len(r[0]) + [0] * (width - len(r[0])) for r in rows])
    labs = torch.tensor([r[1] + [-100] * (width - len(r[1])) for r in rows])
    return {"input_ids": ids, "attention_mask": mask, "labels": labs}


def count(batch):
    return batch["labels"][:, 1:].ne(-100).sum().clamp_min(1)


def score(model, batch):
    dev = next(model.parameters()).device
    ids = batch["input_ids"].to(dev)
    mask = batch["attention_mask"].to(dev)
    labs = batch["labels"].to(dev)
    out = model(input_ids=ids, attention_mask=mask)
    logits = out.logits[:, :-1].float()
    tgt = labs[:, 1:]
    keep = tgt.ne(-100)
    safe = tgt.masked_fill(~keep, 0)
    nll = F.cross_entropy(logits.transpose(1, 2), safe, reduction="none")
    return -(nll * keep).sum(dim=1)


def sftloss(model, tok, qs, ans, limit):
    batch = collate(tok, list(zip(qs, ans)), limit)
    lp = score(model, batch)
    return -lp.sum() / count(batch).to(lp.device)


def dpobatch(tok, qs, good, bad, limit):
    return {
        "chosen": collate(tok, list(zip(qs, good)), limit),
        "rejected": collate(tok, list(zip(qs, bad)), limit),
    }


def dpoloss(pc, pr, rc, rr, beta):
    gap = beta * ((pc - pr) - (rc - rr))
    return -F.logsigmoid(gap).mean()


def dpofull(policy, ref, tok, qs, good, bad, beta, limit):
    both = dpobatch(tok, qs, good, bad, limit)
    pc = score(policy, both["chosen"])
    pr = score(policy, both["rejected"])
    with torch.no_grad():
        rc = score(ref, both["chosen"])
        rr = score(ref, both["rejected"])
    return dpoloss(pc, pr, rc.to(pc.device), rr.to(pc.device), beta)
