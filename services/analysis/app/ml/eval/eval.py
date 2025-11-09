import argparse, yaml, json
from app.ml.eval.winrate_tournament import tournament
from app.ml.eval.eval_metrics import winrate, harmful_rate
from app.ml.eval.safety_checks import run_safety

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    prompts = [json.loads(l)["prompt"] for l in open(cfg["eval_prompts_file"])]
    pols = cfg["policy_paths"]
    ref = cfg["reference_path"]
    scores = tournament(pols, ref, prompts[: cfg["tournament_samples"]])
    total = len(prompts[: cfg["tournament_samples"]])
    for p, s in scores.items():
        print(f"{p}: winrate vs ref = {s/ max(total,1):.2f}")

    adv = [l.strip() for l in open(cfg["safety"]["adversarial_file"]) if l.strip()]
    harms, n = run_safety(ref, adv, cfg["safety"]["max_new_tokens"])
    print(f"Safety harmful rate (lower better): {harms/max(n,1):.2f}")

if __name__ == "__main__":
    main()


# Stub classes for API compatibility
class ModelEvaluator:
    """Placeholder model evaluator"""
    def __init__(self, model_path: str = None):
        self.model_path = model_path
    
    def evaluate(self, data_path: str, **kwargs):
        return {"status": "not_implemented", "message": "ModelEvaluator not yet implemented"}

class SafetyChecker:
    """Placeholder safety checker"""
    def __init__(self):
        pass
    
    def check(self, **kwargs):
        return {"status": "not_implemented", "message": "SafetyChecker not yet implemented"}
