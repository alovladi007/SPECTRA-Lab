import json, argparse, random, pathlib

RUBRIC = (pathlib.Path(__file__).resolve().parents[1] / "src" / "prompts" / "reward_rubric.md").read_text()

TEMPLATE = """You are a strict expert judge. Given a user prompt and two candidate answers,
pick the better one following this rubric:
{rubric}

Return JSON: {{"winner": "A" or "B", "justification": "...", "confidence": 0-1}}.

User prompt:
{prompt}

Candidate A:
{a}

Candidate B:
{b}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, required=True, help="JSONL with prompt, chosen, rejected (order will be shuffled)")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    out = []
    with open(args.pairs) as f:
        for line in f:
            ex = json.loads(line)
            A, B = ex["chosen"], ex["rejected"]
            if random.random() < 0.5:
                A, B = B, A
                flipped = True
            else:
                flipped = False
            prompt = TEMPLATE.format(rubric=RUBRIC, prompt=ex["prompt"], a=A, b=B)
            out.append({"prompt": prompt, "meta": {"flipped": flipped}})

    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("Wrote judge prompts to", args.out)

if __name__ == "__main__":
    main()
