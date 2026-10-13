import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/lora")
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--accum", type=int, default=4)
args = ap.parse_args()

name = "HuggingFaceTB/SmolLM2-360M-Instruct"
tok = AutoTokenizer.from_pretrained(name)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)

raw = json.load(open("data/persona_lora/ramsay_style_preferences.json"))
rows = []
for r in raw:
    rows.append(
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        }
    )
ds = Dataset.from_list(rows)

lcfg = LoraConfig(
    r=args.rank,
    lora_alpha=2 * args.rank,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
cfg = DPOConfig(
    output_dir=args.out,
    per_device_train_batch_size=args.bs,
    gradient_accumulation_steps=args.accum,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    warmup_ratio=0.1,
    beta=0.1,
    max_length=512,
    logging_steps=2,
    save_strategy="no",
    report_to="none",
    fp16=True,
    seed=42,
)
trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lcfg)
trainer.train()
trainer.save_model(args.out)
json.dump(trainer.state.log_history, open(os.path.join(args.out, "history.json"), "w"), indent=1)
