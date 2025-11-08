import argparse, yaml, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from src.utils.data_utils import pairwise_rm_batch
from src.utils.reward_utils import pairwise_loss
from src.utils.logging_utils import log_section, info

class PairwiseRMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        chosen = {k:v.to(model.device) for k,v in inputs["chosen"].items()}
        rejected = {k:v.to(model.device) for k,v in inputs["rejected"].items()}
        c_scores = model(**chosen).logits.squeeze(-1)
        r_scores = model(**rejected).logits.squeeze(-1)
        loss = pairwise_loss(c_scores, r_scores)
        return (loss, {"c_scores": c_scores, "r_scores": r_scores}) if return_outputs else loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    log_section("Reward Model")
    tok = AutoTokenizer.from_pretrained(cfg["sft_model_path"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(cfg["sft_model_path"], num_labels=1)

    ds = load_dataset("json", data_files={"train": cfg["prefs_train_file"], "val": cfg["prefs_val_file"]})
    def map_fn(ex):
        chosen, rejected = pairwise_rm_batch([ex], tok)
        return {"chosen": chosen, "rejected": rejected}
    ds_tok = ds.map(map_fn, remove_columns=ds["train"].column_names)

    args_tr = TrainingArguments(
        output_dir=cfg["rm_output_dir"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=1,
        fp16=True,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        report_to="none"
    )

    trainer = PairwiseRMTrainer(model=model, args=args_tr, train_dataset=ds_tok["train"], eval_dataset=ds_tok["val"])
    trainer.train()
    trainer.save_model(cfg["rm_output_dir"])
    info(f"Saved RM to {cfg['rm_output_dir']}")

if __name__ == "__main__":
    main()
