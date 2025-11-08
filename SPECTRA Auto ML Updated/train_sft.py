import argparse, yaml, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from src.utils.data_utils import sft_collate
from src.utils.logging_utils import log_section, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    log_section("SFT")
    model_id = cfg["model_id"]
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)
    data_files = {"train": cfg["train_file"], "val": cfg["val_file"]}
    ds = load_dataset("json", data_files=data_files)

    extra_kwargs = {}
    if cfg.get('deepspeed'): extra_kwargs['deepspeed'] = cfg['deepspeed']
    if cfg.get('fsdp'): extra_kwargs['fsdp'] = cfg['fsdp']
    if cfg.get('fsdp_config'): extra_kwargs['fsdp_config'] = cfg['fsdp_config']
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"],
        fp16=cfg.get("fp16", False),
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        tokenizer=tok,
        data_collator=lambda ex: sft_collate(ex, tok, cfg["max_length"]),
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    info(f"Saved SFT policy to {cfg['output_dir']}")

if __name__ == "__main__":
    main()
