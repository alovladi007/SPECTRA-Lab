# Human Labeling Guidelines (Preferences)

## Goal
Given a prompt and two candidate responses, pick the **better** one for:
- **Helpfulness & accuracy** (correct, complete, cites when needed)
- **Clarity** (structure, readability)
- **Safety** (avoids harmful or disallowed content; uses cautious language)
- **Domain rigor** (electronics/photonics/biomed specifics)

## Steps
1. Read the prompt and both answers fully.
2. Disqualify any answer with **unsafe** or **obviously wrong** content.
3. Prefer answers with explicit **reasoning**, units, equations where relevant.
4. If both are similar quality, choose the one with **fewer hallucinations** or that **admits uncertainty**.

## Edge cases
- Ties allowed? Prefer **chosen** but if truly equal, mark `meta.tie=true` (optional).
- If the question is unsafe, prefer an answer that **refuses** and **redirects** safely.

## Examples
See `src/data/schemas/prefs_example.jsonl`.
