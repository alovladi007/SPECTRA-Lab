# AI Feedback (AIF) Judges — Usage Guide

You can bootstrap preferences with an **LLM judge** before human review.

## Judge prompt structure
- Provide: the user prompt, candidate A, candidate B, and a **rubric** (see `src/prompts/reward_rubric.md`).
- Ask the judge to pick **A or B**, plus a brief justification and a confidence score.

## Safety
- Never use the same base model you are optimizing as the only judge.
- For production quality, **sample multiple judges** and **aggregate** (majority vote or score).

## Leakage checks
- Randomize candidate order and blind the judge to model IDs.
- Audit judge consistency with spot human checks.
