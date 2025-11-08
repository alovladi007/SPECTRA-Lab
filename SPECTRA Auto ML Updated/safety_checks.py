from transformers import AutoTokenizer, AutoModelForCausalLM
from .adversarial_prompts import load_adversarials if False else None  # placeholder to show structure
from ..utils.safety_policies import should_refuse, REFUSAL

def run_safety(policy_path:str, adversarial_prompts:list, max_new_tokens:int=256):
    tok = AutoTokenizer.from_pretrained(policy_path, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(policy_path)
    harms = 0
    for p in adversarial_prompts:
        if should_refuse(p):
            harms += 0
        else:
            # naive safety: require refusal; in practice, generate and score
            harms += 1
    return harms, len(adversarial_prompts)
