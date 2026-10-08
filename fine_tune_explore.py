import os, json, torch
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_sequence, render_options, QTYPES

MODEL_ID = "convaiinnovations/laya"
print(f"Fetching tokenizer and config from {MODEL_ID}...")
model_dir = snapshot_download(MODEL_ID)
_fix_tokenizer_config(model_dir)

tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
    cfg = json.load(f)

print("Downloading LocalLLaMA/typed-decisions (train split)...")
ds_train = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")


def build_readable_item(state, q, gold_q):
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "choice":
        keys = list(crit.keys())
        target_labels = keys
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    elif t == "noul":
        target_labels = ["false", "true"]
        target = [gold_q["probabilities"].get("false", 0.5), gold_q["probabilities"].get("true", 0.5)]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target_labels = [str(i) for i in range(n_levels)]
        target = [gold_q["probabilities"].get(str(i), 0.0) for i in range(n_levels)]
    else:
        return None  # unknown question type

    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))
    options = render_options({"t": t, "crit": crit})

    # Still build the token sequence so we only keep items the model can actually use
    seq, markers = build_sequence(
        tok, state, {"t": t, "ins": q["instructions"], "crit": crit},
        cfg["max_len"], cfg["head_max_len"],
    )
    if len(markers) != len(options):
        return None

    return {
        # Raw inputs, exactly as you'd supply them in your own data
        "state": state,
        "instructions": q["instructions"],
        "criteria": crit,
        "options": options,
        # What the model actually sees, as text
        "text": tok.decode(seq, skip_special_tokens=False),
        "tokens": tok.convert_ids_to_tokens(seq),
        # Each option marker: its position and the token string at that spot
        "markers": [{"position": m, "token": tok.convert_ids_to_tokens(seq[m])} for m in markers],
        # Labels in readable form
        "qtype": t,
        "qtype_id": QTYPES[t],
        "target": dict(zip(target_labels, target)),
        "label": target_labels[label],
    }


items = []
for row in ds_train:
    state = json.loads(row["state"])
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    for qid, q in questions.items():
        if qid in gold:
            it = build_readable_item(state, q, gold[qid])
            if it:
                items.append(it)

print(f"Preprocessed {len(items)} training sequences across {len(ds_train)} cases.")

# Inspect one example
print(json.dumps(items[0], indent=2, default=str)[:3000])