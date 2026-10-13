import torch
import dpolib as lib
from tests.toy_lm import ToyCausalLM, ToyTokenizer


def test_truncation_cuts_eos():
    tok = ToyTokenizer()
    b = lib.collate(tok, [("abcd", "efghijkl")], 6)
    assert b["input_ids"].shape == (1, 6)
    assert b["labels"][0, :4].eq(-100).all()
    assert b["labels"][0, 4:].ne(-100).all()
    assert b["input_ids"][0, -1] != tok.eos_token_id


def test_eos_kept_when_it_fits():
    tok = ToyTokenizer()
    b = lib.collate(tok, [("ab", "c")], 10)
    assert b["input_ids"].shape == (1, 4)
    assert b["input_ids"][0, -1] == tok.eos_token_id
    assert b["labels"][0, -1] == tok.eos_token_id


def test_row_order_survives():
    tok = ToyTokenizer()
    pairs = [("aa", "b"), ("cccc", "dddd"), ("e", "f")]
    b = lib.collate(tok, pairs, 20)
    for i, (q, a) in enumerate(pairs):
        ids, labs = lib.encode(tok, q, a, 20)
        assert b["input_ids"][i, : len(ids)].tolist() == ids
        assert b["labels"][i, : len(labs)].tolist() == labs


def test_padding_does_not_change_scores():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, 0.5)
    alone = lib.score(model, lib.collate(tok, [("ab", "cd")], 40))
    mixed = lib.score(model, lib.collate(tok, [("ab", "cd"), ("a much longer prompt", "and a long answer")], 40))
    assert torch.allclose(alone[0], mixed[0], atol=1e-6)


def test_dpo_formula_by_hand():
    pc = torch.tensor([-3.0, -7.0])
    pr = torch.tensor([-5.0, -6.0])
    rc = torch.tensor([-4.0, -7.5])
    rr = torch.tensor([-4.5, -6.5])
    beta = 0.1
    want = -torch.log(torch.sigmoid(beta * ((pc - pr) - (rc - rr)))).mean()
    got = lib.dpoloss(pc, pr, rc, rr, beta)
    assert torch.allclose(got, want, atol=1e-6)


def test_one_step_lowers_dpo_loss():
    tok = ToyTokenizer()
    pol = ToyCausalLM(tok.vocab_size)
    ref = ToyCausalLM(tok.vocab_size)
    args = (tok, ["Q: "], ["good"], ["bad"], 0.1, 20)
    before = lib.dpofull(pol, ref, *args)
    before.backward()
    with torch.no_grad():
        pol.logit_bias -= 0.5 * pol.logit_bias.grad
    after = lib.dpofull(pol, ref, *args)
    assert after.item() < before.item()
