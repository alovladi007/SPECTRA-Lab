# Simple convenience targets
.PHONY: init sft rm ppo dpo eval docker

init:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

sft:
	python src/train_sft.py --config configs/sft.yaml

rm:
	python src/train_rm.py --config configs/rlhf.yaml

ppo:
	python src/train_ppo.py --config configs/rlhf.yaml

dpo:
	python src/train_dpo.py --config configs/rlhf.yaml

eval:
	python src/eval/eval.py --config configs/eval.yaml

docker:
	docker build -t rlhf-base:latest .
