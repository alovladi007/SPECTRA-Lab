# RLHF Base Repo (Text Policy) — Electronics/Photonics/Biomed

This repository gives you a **ready-to-run** pipeline to:
- Fine-tune a base LLM with **SFT**,
- Train a **Reward Model (RM)** on human or AI feedback,
- Optimize the policy with **PPO** (classic RLHF) or **DPO** (pairwise preference without explicit RM),
- **Evaluate** via win-rate tournaments and **safety checks**,
- **Bootstrap** your domain with tailored **prompt packs** (electronics, photonics, biomedical).

> Tested with Python 3.10+. You’ll need a CUDA GPU for training.

## 🚀 NEW: AutoML for Semiconductor Manufacturing

**Automated Machine Learning capabilities now available!**

This repository now includes comprehensive AutoML features specifically designed for semiconductor manufacturing processes:

### AutoML Features
- 🤖 **Auto Model Selection** - Automatically find the best algorithm for your data
- 🎯 **Hyperparameter Tuning** - Optimize model parameters automatically using Bayesian optimization
- 🏗️ **Neural Architecture Search** - Design optimal neural networks automatically

### Quick Start with AutoML

```bash
# Run quick AutoML pipeline (5-10 minutes)
python src/automl/train_automl.py --config configs/automl/automl_quickstart.yaml

# Run full AutoML pipeline (30-60 minutes)
python src/automl/train_automl.py --config configs/automl/automl_full.yaml
```

**📖 See [AUTOML_README.md](AUTOML_README.md) for complete AutoML documentation**


---

## Quickstart

```bash
# 0) Create env
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# 1) Install deps
pip install -r requirements.txt

# 2) (Optional) Login to HF if you want to push models
# huggingface-cli login

# 3) Prepare a small dummy dataset (creates toy JSONLs)
python scripts/prepare_data.py

# 4) Run SFT
python src/train_sft.py --config configs/sft.yaml

# 5) Train Reward Model (pairwise preferences)
python src/train_rm.py --config configs/rlhf.yaml

# 6) PPO RLHF (uses the trained RM)
python src/train_ppo.py --config configs/rlhf.yaml

# (Alternative) DPO (no explicit RM required)
python src/train_dpo.py --config configs/rlhf.yaml

# 7) Evaluate (win-rate & safety)
python src/eval/eval.py --config configs/eval.yaml
```

### Minimal data shapes

- **SFT**: `{"prompt": "...", "response": "..."}` per line (JSONL).
- **Preferences**: `{"prompt": "...", "chosen": "...", "rejected": "..."}` per line (JSONL).

See `src/data/schemas/` for JSON Schemas and labeling guides.

---

## Repo layout

```
.
├── configs/
│   ├── sft.yaml
│   ├── rlhf.yaml
│   └── eval.yaml
├── src/
│   ├── train_sft.py
│   ├── train_rm.py
│   ├── train_ppo.py
│   ├── train_dpo.py
│   ├── utils/
│   │   ├── data_utils.py
│   │   ├── reward_utils.py
│   │   ├── logging_utils.py
│   │   ├── safety_policies.py
│   │   └── eval_metrics.py
│   ├── eval/
│   │   ├── eval.py
│   │   ├── winrate_tournament.py
│   │   ├── safety_checks.py
│   │   └── adversarial_prompts.txt
│   ├── prompts/
│   │   ├── system_electronics.txt
│   │   ├── system_photonics.txt
│   │   ├── system_biomed.txt
│   │   └── reward_rubric.md
│   └── data/
│       └── schemas/
│           ├── preference_pair.schema.json
│           ├── sft_example.jsonl
│           ├── prefs_example.jsonl
│           ├── human_labeling_guidelines.md
│           └── aif_judges_guidelines.md
├── scripts/
│   ├── prepare_data.py
│   ├── sample_prompts.jsonl
│   └── generate_judgments_aif.py
├── requirements.txt
├── Dockerfile
├── Makefile
└── .gitignore
```

---

## Domain-tailored prompting

We include **system prompts** to bias the model toward **electronics**, **photonics**, and **biomed** expertise. You can mix them in SFT and RLHF by prepending the appropriate `system_*` file to each prompt.

---

## Safety & governance

- Rule-based **refusals** for disallowed or high-risk requests.
- **Hallucination guards** (ask-for-citation, uncertainty prompts).
- Adversarial prompts for red-teaming.
- Metrics: refusal precision/recall, harmful content rate, jailbreak rate.

---

## License

MIT (for this scaffold). Verify licenses of any datasets/models you train.


## Multi-GPU with DeepSpeed or FSDP

This repo ships ready-made configs for **DeepSpeed ZeRO-3** (`configs/ds_zero3.json`) and **PyTorch FSDP** (`configs/fsdp_config.json`).

### DeepSpeed
- Already enabled in `configs/sft.yaml` via `deepspeed: configs/ds_zero3.json`.
- Launch example (2 GPUs):
```bash
accelerate launch --num_processes=2 src/train_sft.py --config configs/sft.yaml
```

### FSDP
- Comment out `deepspeed` and set:
```yaml
fsdp: full_shard auto_wrap
fsdp_config: configs/fsdp_config.json
```
- Then launch with Accelerate (2 GPUs):
```bash
accelerate launch --num_processes=2 src/train_sft.py --config configs/sft.yaml
```

> Tip: You can use `accelerate config` once to create a default multi-GPU config. PPO/DPO can also be launched via `accelerate`.
