import argparse, yaml, torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from datasets import load_dataset
from src.utils.logging_utils import log_section, info
from src.utils.safety_policies import should_refuse
from transformers import pipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    log_section("PPO RLHF")
    tok = AutoTokenizer.from_pretrained(cfg["sft_model_path"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(cfg["sft_model_path"])
    pconf = PPOConfig(batch_size=cfg["ppo"]["batch_size"], learning_rate=cfg["ppo"]["learning_rate"],
                      ppo_epochs=cfg["ppo"]["ppo_epochs"], target_kl=cfg["ppo"]["target_kl"])

    # Load prompts from prefs train (or your own prompts file)
    ds = load_dataset("json", data_files={"train": cfg["prefs_train_file"]})["train"]
    prompts = [ex["prompt"] for ex in ds]

    # Reward model (sequence classifier head)
    rm_pipe = pipeline("text-classification", model=cfg["rm_output_dir"], tokenizer=tok, device_map="auto", return_all_scores=False)

    ppo = PPOTrainer(config=pconf, model=policy, tokenizer=tok)
    max_new = cfg["ppo"]["gen_max_new_tokens"]

    for i, prompt in enumerate(prompts):
        if should_refuse(prompt):
            # Encourage refusals by giving small positive reward for safe refusal
            input_ids = tok(prompt, return_tensors="pt").input_ids.to(policy.pretrained_model.device)
            output_ids = policy.generate(input_ids, max_new_tokens=32)
            ppo.step([prompt], [tok.decode(output_ids[0], skip_special_tokens=True)], [0.1])
            continue

        input_ids = tok(prompt, return_tensors="pt").input_ids.to(policy.pretrained_model.device)
        gen_ids = policy.generate(input_ids, max_new_tokens=max_new, do_sample=True, top_p=0.9, temperature=0.7)
        response = tok.decode(gen_ids[0], skip_special_tokens=True)

        score = rm_pipe(f"### Prompt:\n{prompt}\n### Response:\n{response}")[0]["score"]
        reward = float(score)  # simplistic; consider normalization/clipping
        ppo.step([prompt], [response], [reward])

        if (i+1) % 10 == 0:
            info(f"Step {i+1}: reward={reward:.3f}")

    ppo.model.save_pretrained(cfg["ppo_output_dir"])
    info(f"Saved PPO policy to {cfg['ppo_output_dir']}")

if __name__ == "__main__":
    main()
