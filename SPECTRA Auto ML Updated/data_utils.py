import json, random, os
from typing import List, Dict, Iterable

def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]

def yield_jsonl(path: str) -> Iterable[Dict]:
    with open(path) as f:
        for l in f:
            yield json.loads(l)

def sft_collate(examples, tokenizer, max_length=2048):
    inputs = []
    for ex in examples:
        text = f"### Prompt:\n{ex['prompt']}\n### Response:\n{ex['response']}"
        inputs.append(text)
    tok = tokenizer(inputs, truncation=True, max_length=max_length, padding=True)
    tok["labels"] = tok["input_ids"].copy()
    return tok

def pairwise_rm_batch(pairs, tokenizer, max_length=2048):
    # returns tokenized (prompt+response) for chosen and rejected
    chosen_texts, rejected_texts = [], []
    for p in pairs:
        chosen_texts.append(f"### Prompt:\n{p['prompt']}\n### Response:\n{p['chosen']}")
        rejected_texts.append(f"""### Prompt:\n{p['prompt']}\n### Response:\n{p['rejected']}""")
    chosen = tokenizer(chosen_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    rejected = tokenizer(rejected_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return chosen, rejected
