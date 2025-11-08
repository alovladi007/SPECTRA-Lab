import argparse, yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig, AutoModelForCausalLMWithValueHead
from src.utils.logging_utils import log_section, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    log_section("DPO")
    tok = AutoTokenizer.from_pretrained(cfg["sft_model_path"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(cfg["sft_model_path"])

    ds = load_dataset("json", data_files={"train": cfg["prefs_train_file"], "val": cfg["prefs_val_file"]})
    dpo_cfg = DPOConfig(beta=cfg["dpo"]["beta"], learning_rate=cfg["dpo"]["lr"],
                        max_length=1024, max_target_length=256,
                        per_device_train_batch_size=cfg["dpo"]["per_device_train_batch_size"],
                        num_train_epochs=cfg["dpo"]["epochs"])

    trainer = DPOTrainer(model, tok, **dpo_cfg.to_dict(), train_dataset=ds["train"], eval_dataset=ds["val"])
    trainer.train()
    trainer.save_model(cfg["dpo_output_dir"])
    info(f"Saved DPO policy to {cfg['dpo_output_dir']}")

if __name__ == "__main__":
    main()
